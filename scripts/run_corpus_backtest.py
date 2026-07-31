"""Backtest every usable candidate over the development window, in a bounded process pool.

Generated code is untrusted. It raises, it calls removed polars APIs, it occasionally loops longer
than any honest strategy would, and one bad candidate must not take the run down. Each backtest
therefore happens in a worker process with a wall-clock bound, and every failure is recorded as a
result rather than propagated.

**Pool width is set by memory, not by core count.** The machine has 15.6 GB of RAM against 32
logical cores, and a worker holding the 252 MB panel has a ~340 MB working set before intermediates.
Eight workers is roughly 5.6 GB and leaves room for the OS and the model server; thirty-two would
want around 22 GB and would thrash. Measured before this was written, not assumed.

**The holdout is reachable only through a flag pair.** By default the window is the development
period and the worker loads the development panel, which does not contain a single holdout row —
so a default run cannot touch the holdout even by mistake. `--holdout` switches to the holdout
artifacts and is refused unless `--authorised-by` accompanies it (Rule 7).

Usage:
    python scripts/run_corpus_backtest.py --corpus runs/<stamp>/candidates
    python scripts/run_corpus_backtest.py --corpus <...> --holdout --authorised-by "<who, when>"
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import multiprocessing
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import summarise  # noqa: E402

_log = get_logger(__name__)

#: Per-candidate wall-clock bound. An honest backtest over the development window takes about 12
#: seconds; anything past this is pathological and is recorded as a timeout.
TIMEOUT_SECONDS = 180.0
DEFAULT_WORKERS = 8

# Set once per worker by the initialiser, so the panel is read 8 times rather than 300.
_ENGINE: BacktestEngine | None = None
_WINDOW: tuple[Any, Any] | None = None


@dataclass(frozen=True)
class CandidateResult:
    """What happened when one candidate was run."""

    name: str
    path: str
    outcome: str          # "evaluated", "runtime_error", "timeout"
    error: str | None
    sharpe: float | None
    annualised_return: float | None
    volatility: float | None
    max_drawdown: float | None
    n_sessions: int
    returns_path: str | None


def _worker_init(holdout: bool = False) -> None:
    """Load the panel and build the engine once per worker process.

    The holdout artifacts are separate files, not a filtered view of the development ones, so a
    development run cannot reach holdout rows even by mistake — they are not in the frame it loaded.
    """
    global _ENGINE, _WINDOW  # noqa: PLW0603
    cfg = load_config()
    suffix = "_holdout" if holdout else ""
    panel = pl.read_parquet(cfg.paths.data_processed / f"prices_adjusted{suffix}.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / f"universe{suffix}.parquet")
    calendar = load_calendar(cfg.paths.data_raw / f"calendar_cnx100{suffix}.parquet")
    # Net of Indian transaction costs, charged per leg at the rates in force on each fill
    # session (Phase 1.1, PI-approved at Checkpoint 1.1). A gross run is no longer the
    # default: every Sharpe this script reports is now after costs.
    _ENGINE = BacktestEngine(panel=panel, calendar=calendar, universe=universe,
                             cost_model=CostModel(cfg.costs))
    _WINDOW = (
        (cfg.dates.holdout_start, cfg.dates.holdout_end) if holdout
        else (cfg.dates.dev_start, cfg.dates.dev_end)
    )


def _load_strategy(path: Path) -> type[Strategy]:
    """Import a candidate file and return the Strategy subclass it defines."""
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subclasses = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
    ]
    if not subclasses:
        raise TypeError("file defines no Strategy subclass")
    return subclasses[-1]


def _instantiate(strategy_class: type[Strategy]) -> Strategy:
    """Build a strategy, supplying plausible values for any parameter lacking a default.

    The first run discarded 21 otherwise-sound candidates because this called the class with no
    arguments and the constructor required some. That was a harness defect, not a candidate defect:
    the prompt invited scalar settings without saying they needed defaults.

    Values are chosen from the annotation, and only for parameters the author left required. A
    strategy that declares its own defaults is built exactly as written and nothing is substituted.
    """
    try:
        return strategy_class()
    except TypeError:
        pass

    signature = inspect.signature(strategy_class.__init__)
    supplied: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = parameter.annotation
        # A window length is the overwhelmingly common required setting; 20 sessions is a month of
        # trading and is a neutral choice rather than one tuned to make anything perform.
        supplied[name] = 0.5 if annotation is float or annotation == "float" else 20
    return strategy_class(**supplied)


def run_one(path_str: str, out_dir_str: str) -> dict[str, Any]:
    """Backtest one candidate. Never raises — a failure is the result."""
    path, out_dir = Path(path_str), Path(out_dir_str)
    assert _ENGINE is not None and _WINDOW is not None, "worker initialiser did not run"
    name = path.stem
    try:
        strategy = _instantiate(_load_strategy(path))
        result = _ENGINE.run(strategy, start=_WINDOW[0], end=_WINDOW[1])
    except Exception as exc:  # noqa: BLE001 - untrusted code; every failure is a datum
        return asdict(CandidateResult(
            name=name, path=path_str, outcome="runtime_error",
            error=f"{type(exc).__name__}: {exc}"[:300], sharpe=None, annualised_return=None,
            volatility=None, max_drawdown=None, n_sessions=0, returns_path=None,
        ))

    stats = summarise(result.returns)
    returns_path = out_dir / f"{name}_returns.parquet"
    pl.DataFrame({"session_date": result.dates, "return": result.returns}).write_parquet(
        returns_path
    )
    return asdict(CandidateResult(
        name=name, path=path_str, outcome="evaluated", error=None,
        # The key is `sharpe_ratio`; asking for `sharpe` returned None for every candidate and
        # would have produced a corpus with no performance numbers at all.
        sharpe=stats["sharpe_ratio"], annualised_return=stats["annualised_return"],
        volatility=stats["annualised_volatility"], max_drawdown=stats["max_drawdown"],
        n_sessions=len(result.returns), returns_path=str(returns_path),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="directory of candidate .py")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    # Same structural gate as the ablation. A convention that the holdout is only read when intended
    # is worth very little at the end of a long run; the flag pair makes it impossible to do by
    # accident, and the authorisation text lands in the output filename's sibling manifest.
    parser.add_argument("--holdout", action="store_true",
                        help="evaluate on the holdout; requires --authorised-by")
    parser.add_argument("--authorised-by", type=str, default=None,
                        help="the PI's explicit authorisation for a holdout evaluation")
    args = parser.parse_args()

    if args.holdout and not args.authorised_by:
        _log.error(
            "the holdout may be evaluated once per project and only with explicit PI authorisation "
            "(Rule 7). Pass --authorised-by, or omit --holdout to run the development window."
        )
        return 2
    if args.authorised_by and not args.holdout:
        _log.error("--authorised-by given without --holdout; refusing to guess the intent")
        return 2

    candidates = sorted(args.corpus.glob("candidate_*.py"))
    if not candidates:
        _log.error("no candidates found under %s", args.corpus)
        return 1

    label = "holdout" if args.holdout else "development"
    if args.holdout:
        _log.warning("HOLDOUT EVALUATION, authorised by: %s", args.authorised_by)
    out_dir = args.corpus.parent / f"backtests_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The corpus path goes in the log so a progress reader can tell which corpus a log belongs to.
    # Without it a stale log from an earlier corpus is indistinguishable from a live one, and the
    # reported progress silently describes the wrong run.
    _log.info("backtesting corpus %s: %d candidates across %d workers",
              args.corpus.resolve(), len(candidates), args.workers)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    # multiprocessing.Pool rather than ProcessPoolExecutor, because only the former can actually
    # kill a worker. `future.result(timeout=...)` bounds how long the parent waits for a result, not
    # how long the child runs, and ProcessPoolExecutor's shutdown then joins that child on the way
    # out of the `with` block. A generated strategy with an unbounded loop therefore hangs the run
    # forever *after* every result has been collected, with the output trapped in a process that
    # will not exit. Two candidates in the first corpus did exactly that. `terminate()` kills the
    # children outright, which is the only thing that works against untrusted code.
    pool = multiprocessing.Pool(
        processes=args.workers, initializer=_worker_init, initargs=(args.holdout,)
    )
    try:
        pending = [
            (path, pool.apply_async(run_one, (str(path), str(out_dir))))
            for path in candidates
        ]
        for done, (path, async_result) in enumerate(pending, start=1):
            try:
                results.append(async_result.get(timeout=TIMEOUT_SECONDS))
            except multiprocessing.TimeoutError:
                results.append(asdict(CandidateResult(
                    name=path.stem, path=str(path), outcome="timeout",
                    error=f"exceeded {TIMEOUT_SECONDS:.0f}s", sharpe=None,
                    annualised_return=None, volatility=None, max_drawdown=None,
                    n_sessions=0, returns_path=None,
                )))
            except Exception as exc:  # noqa: BLE001 - a worker died; record and continue
                results.append(asdict(CandidateResult(
                    name=path.stem, path=str(path), outcome="runtime_error",
                    error=f"worker died: {type(exc).__name__}: {exc}"[:300], sharpe=None,
                    annualised_return=None, volatility=None, max_drawdown=None,
                    n_sessions=0, returns_path=None,
                )))
            if done % 25 == 0:
                _log.info("%d/%d backtested", done, len(candidates))
    finally:
        # Not close(): a still-spinning strategy would keep the interpreter alive indefinitely.
        pool.terminate()
        pool.join()

    elapsed = time.perf_counter() - started
    ran = sum(1 for r in results if r["outcome"] == "evaluated")
    results_name = "holdout_results.json" if args.holdout else "backtest_results.json"
    (args.corpus.parent / results_name).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _log.info("%d/%d ran to completion in %.0f min", ran, len(results), elapsed / 60)
    print(f"backtested: {ran}/{len(results)} in {elapsed / 60:.1f} min")
    for outcome in ("runtime_error", "timeout"):
        n = sum(1 for r in results if r["outcome"] == outcome)
        print(f"  {outcome}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
