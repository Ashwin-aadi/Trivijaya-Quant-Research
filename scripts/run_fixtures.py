"""Run the leaky and clean fixtures over real data and report their statistics.

Two jobs. The first is to confirm the three deliberate cheats produce obviously absurd results —
they are the positive controls for the leakage auditor, and a cheat that does not look ridiculous
means the engine is failing to let it cheat, which would invalidate the whole exercise.

The second matters just as much and is easy to skip: confirming each fixture is absurd *for its
intended reason*. Real artifacts exist in the price series — an unresolved capital change, two
demergers, a pair of genuine crashes — and a strategy that happened to trade one of those dates
could post a spectacular number without its intended cheat contributing anything. Every run is
therefore re-scored with all registered artifact dates neutralised. If a fixture's result barely
moves, the cheat is doing the work; if it collapses, the number was a data scar and the fixture is
not a valid control.

Usage:
    python scripts/run_fixtures.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.data.artifacts import REGISTER_FILENAME  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import sharpe_ratio, summarise  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "fixtures"))

from tests.fixtures.leaky.leak_full_sample_scaler import LeakFullSampleScaler  # noqa: E402
from tests.fixtures.leaky.leak_future_return import LeakFutureReturn  # noqa: E402
from tests.fixtures.leaky.leak_survivorship import LeakSurvivorship  # noqa: E402

_log = get_logger("run_fixtures")


def _artifact_free_panel(panel: pl.DataFrame, register: pl.DataFrame) -> pl.DataFrame:
    """Drop every symbol-date covered by the artifact register.

    Used to re-score a strategy on data known to be clean. Removing rows shortens the series
    rather than distorting it, which is the honest way to ask "does this result survive without
    the questionable points?".
    """
    if register.is_empty():
        return panel
    keep = panel
    for row in register.iter_rows(named=True):
        keep = keep.filter(
            ~(
                (pl.col("symbol") == row["symbol"])
                & (pl.col("session_date") >= row["start_date"])
                & (pl.col("session_date") <= row["end_date"])
            )
        )
    return keep


def _score(
    strategy: Strategy,
    panel: pl.DataFrame,
    calendar: object,
    universe: pl.DataFrame,
    start: date,
    end: date,
    register: pl.DataFrame | None = None,
) -> dict[str, float]:
    engine = BacktestEngine(panel, calendar, universe, artifact_register=register)  # type: ignore[arg-type]
    result = engine.run(strategy, start, end)
    stats = summarise(result.returns)
    stats["flagged_sessions"] = float(len(set(result.flagged_sessions)))
    return stats


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.meta.seed)
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    register = pl.read_parquet(cfg.paths.data_processed / REGISTER_FILENAME)

    start, end = cfg.dates.dev_start, cfg.dates.dev_end
    clean_panel = _artifact_free_panel(panel, register)
    _log.info("panel %d rows; artifact-free panel %d rows", panel.height, clean_panel.height)

    fixtures: list[tuple[str, Strategy]] = [
        ("leak_future_return", LeakFutureReturn(panel)),
        ("leak_full_sample_scaler", LeakFullSampleScaler(panel)),
        ("leak_survivorship", LeakSurvivorship(universe)),
    ]

    with RunManifest(cfg, script="scripts/run_fixtures.py") as run:
        rows: list[dict[str, object]] = []
        for name, strategy in fixtures:
            full = _score(strategy, panel, calendar, universe, start, end, register)
            # Re-score on data with every registered artifact removed. A cheat that is genuinely
            # doing the work survives this; a result that was really a data scar does not.
            clean = _score(strategy, clean_panel, calendar, universe, start, end)
            rows.append(
                {
                    "fixture": name,
                    "sharpe": full["sharpe_ratio"],
                    "sharpe_artifact_free": clean["sharpe_ratio"],
                    "annualised_return": full["annualised_return"],
                    "max_drawdown": full["max_drawdown"],
                    "n_sessions": full["n_sessions"],
                    "flagged_sessions": full["flagged_sessions"],
                }
            )
            _log.info("%s: sharpe %.2f (artifact-free %.2f) over %d sessions",
                      name, full["sharpe_ratio"], clean["sharpe_ratio"], int(full["n_sessions"]))

        table = pl.DataFrame(rows)
        table.write_parquet(cfg.paths.data_processed / "fixture_results.parquet")
        run.note("fixtures_run", len(rows))

    print("\nLEAKY FIXTURES — deliberate cheats, expected to look absurd")
    print(table.to_pandas().to_string(index=False))
    print("\nsharpe_artifact_free re-scores each fixture with all registered artifact dates")
    print("removed. A cheat that is doing the work keeps its number; a data scar does not.")
    return 0


def buy_and_hold_sharpe(returns: list[float]) -> float:
    """Convenience for the checkpoint report."""
    return sharpe_ratio(returns)


if __name__ == "__main__":
    raise SystemExit(main())
