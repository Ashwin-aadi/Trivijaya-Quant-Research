"""The search signal for the arms that close a loop, net of Indian transaction costs.

Only one arm in P4 scores its own intermediate output: the Monte Carlo tree search arm. Everything
it needs is here, in one place, so that what the search optimises is a documented decision rather
than an ad-hoc lambda at the call site.

**Net of costs, by PI ruling of 2026-08-04.** Every headline in this lab is net. A search bred
against gross Sharpe would optimise toward a number nothing else in the repository reports, and
would breed strategies that die at the cost model. The cost of the ruling is stated rather than
hidden: this arm receives cost information no other arm receives, so part of any advantage it shows
may be cost-optimisation rather than search. That confound is pre-registered.

**The frozen stack is not reachable from here.** The fitness is a backtest and nothing else — no
auditor layer, no fragility model, no capacity model. An arm that selected on the instrument
measuring it would have decided its own result.

**The holdout is not reachable from here either, structurally.** This module loads the development
panel and has no flag, argument or code path that could load anything else. That is deliberate: a
convention would not survive a long unattended run, and a search loop is the single worst place in
the repository for holdout contamination to hide.

**Untrusted code runs in a killable worker.** Generated strategies raise, and two in P1's corpus
looped without bound. A pool of one worker with a wall-clock bound is the only arrangement that can
actually terminate them; the worker is rebuilt after a kill.
"""

from __future__ import annotations

import multiprocessing
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.log import get_logger

_log = get_logger(__name__)

#: Scores one strategy's source. None means it could not be scored at all — unusable code, a
#: strategy that never took a position, or one that had to be killed. Higher is fitter.
Fitness = Callable[[str], float | None]

#: Wall-clock bound per scored candidate. An honest development-window backtest takes about 12
#: seconds; past this the candidate is pathological and scores None, exactly as an unrunnable one
#: does. Matches `scripts/run_corpus_backtest.py`, so a strategy the search could score is a
#: strategy the corpus backtest can score too.
TIMEOUT_SECONDS: float = 180.0

_ENGINE: Any = None
_WINDOW: tuple[Any, Any] | None = None


def _worker_init() -> None:
    """Load the development panel and build the engine once per worker process.

    Imports happen inside the function so that importing this module does not pull polars and the
    price panel into every process that merely wants the type.
    """
    global _ENGINE, _WINDOW  # noqa: PLW0603
    import polars as pl

    from src.backtest.engine import BacktestEngine
    from src.common.config import load_config
    from src.costs.india import CostModel
    from src.data.calendar import load_calendar

    cfg = load_config()
    # No suffix, no branch, no flag. The holdout artifacts are separate files and this line cannot
    # name them.
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    _ENGINE = BacktestEngine(
        panel=panel, calendar=calendar, universe=universe, cost_model=CostModel(cfg.costs)
    )
    _WINDOW = (cfg.dates.dev_start, cfg.dates.dev_end)


def _score_source(source: str) -> float | None:
    """Backtest one source string and return its net Sharpe, or None if it cannot be scored.

    Never raises. A strategy that fails to import, fails to run, or takes no position is not an
    error in the search — it is a candidate with no fitness, and the search must be able to say so.
    """
    import importlib.util
    import inspect

    from src.backtest.strategy import Strategy
    from src.eval.metrics import summarise

    assert _ENGINE is not None and _WINDOW is not None, "worker initialiser did not run"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.py"
            path.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("mcts_candidate", path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        subclasses = [
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
        ]
        if not subclasses:
            return None
        strategy_class = subclasses[-1]

        # Same substitution rule as the corpus backtest: a required scalar the author left without
        # a default is supplied, so the search does not discard sound candidates over a harness
        # detail. A strategy declaring its own defaults is built exactly as written.
        try:
            strategy = strategy_class()
        except TypeError:
            signature = inspect.signature(strategy_class.__init__)
            supplied: dict[str, Any] = {}
            for name, parameter in signature.parameters.items():
                if name == "self" or parameter.default is not inspect.Parameter.empty:
                    continue
                if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                    continue
                annotation = parameter.annotation
                supplied[name] = (
                    0.5 if annotation is float or annotation == "float" else 20
                )
            strategy = strategy_class(**supplied)

        result = _ENGINE.run(strategy, start=_WINDOW[0], end=_WINDOW[1])
    except Exception:  # noqa: BLE001 - untrusted code; an unscoreable candidate is a datum
        return None

    if not result.returns:
        return None
    value = summarise(result.returns)["sharpe_ratio"]
    if value is None:
        return None
    score = float(value)
    # NaN compares false against everything and would silently win or lose a max() depending on
    # argument order. An unscoreable candidate must be None so the search treats it as one.
    return None if score != score else score


class NetSharpeFitness:
    """Scores a strategy source by development-window Sharpe, net of Indian transaction costs.

    Callable, so it satisfies the ``Fitness`` protocol the search arm expects. Holds one worker
    process for the life of the arm, because loading the 252 MB panel per candidate would dominate
    the run; the worker is rebuilt whenever a candidate has to be killed.
    """

    def __init__(self, *, timeout_seconds: float = TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        self._pool: multiprocessing.pool.Pool | None = None
        #: Candidates the search asked about, and how many could not be scored. Reported in the
        #: run manifest: a search whose signal was absent most of the time is a search that mostly
        #: did not search, and the paper must be able to say so.
        self.scored = 0
        self.unscoreable = 0
        self.timeouts = 0

    def _ensure_pool(self) -> multiprocessing.pool.Pool:
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=1, initializer=_worker_init)
        return self._pool

    def __call__(self, source: str) -> float | None:
        if not source.strip():
            return None
        self.scored += 1
        pool = self._ensure_pool()
        try:
            value: float | None = pool.apply_async(_score_source, (source,)).get(
                timeout=self.timeout_seconds
            )
        except multiprocessing.TimeoutError:
            self.timeouts += 1
            self.unscoreable += 1
            _log.warning("fitness timed out after %.0fs; rebuilding worker", self.timeout_seconds)
            self.close()
            return None
        except Exception as exc:  # noqa: BLE001 - the worker died; rebuild and carry on
            self.unscoreable += 1
            _log.warning("fitness worker died (%s); rebuilding", type(exc).__name__)
            self.close()
            return None
        if value is None:
            self.unscoreable += 1
        return value

    def stats(self) -> dict[str, int]:
        return {
            "scored": self.scored,
            "unscoreable": self.unscoreable,
            "timeouts": self.timeouts,
        }

    def close(self) -> None:
        """Terminate rather than close: a spinning strategy would keep the interpreter alive."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
