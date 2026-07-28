"""Backtest every usable candidate over the development window, in a bounded process pool.

Generated code is untrusted. It raises, it calls removed polars APIs, it occasionally loops longer
than any honest strategy would, and one bad candidate must not take the run down. Each backtest
therefore happens in a worker process with a wall-clock bound, and every failure is recorded as a
result rather than propagated.

**Pool width is set by memory, not by core count.** The machine has 15.6 GB of RAM against 32
logical cores, and a worker holding the 252 MB panel has a ~340 MB working set before intermediates.
Eight workers is roughly 5.6 GB and leaves room for the OS and the model server; thirty-two would
want around 22 GB and would thrash. Measured before this was written, not assumed.

**The holdout is not touched here.** The window is the development period from `config.yaml` and
nothing in this script can reach past it.

Usage:
    python scripts/run_corpus_backtest.py --corpus runs/<stamp>/candidates
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
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


def _worker_init() -> None:
    """Load the panel and build the engine once per worker process."""
    global _ENGINE, _WINDOW  # noqa: PLW0603
    cfg = load_config()
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    _ENGINE = BacktestEngine(panel=panel, calendar=calendar, universe=universe)
    _WINDOW = (cfg.dates.dev_start, cfg.dates.dev_end)


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


def run_one(path_str: str, out_dir_str: str) -> dict[str, Any]:
    """Backtest one candidate. Never raises — a failure is the result."""
    path, out_dir = Path(path_str), Path(out_dir_str)
    assert _ENGINE is not None and _WINDOW is not None, "worker initialiser did not run"
    name = path.stem
    try:
        strategy = _load_strategy(path)()
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
        sharpe=stats.get("sharpe"), annualised_return=stats.get("annualised_return"),
        volatility=stats.get("annualised_volatility"), max_drawdown=stats.get("max_drawdown"),
        n_sessions=len(result.returns), returns_path=str(returns_path),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="directory of candidate .py")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    candidates = sorted(args.corpus.glob("candidate_*.py"))
    if not candidates:
        _log.error("no candidates found under %s", args.corpus)
        return 1

    out_dir = args.corpus.parent / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info("backtesting %d candidates across %d workers", len(candidates), args.workers)

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        futures = {
            pool.submit(run_one, str(p), str(out_dir)): p for p in candidates
        }
        for done, (future, path) in enumerate(futures.items(), start=1):
            try:
                results.append(future.result(timeout=TIMEOUT_SECONDS))
            except FutureTimeout:
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

    elapsed = time.perf_counter() - started
    ran = sum(1 for r in results if r["outcome"] == "evaluated")
    (args.corpus.parent / "backtest_results.json").write_text(
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
