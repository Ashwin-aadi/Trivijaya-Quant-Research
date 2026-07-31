"""How much of the cost drag is the fixed per-scrip depository charge?

The depository charge is levied per scrip per sell instruction and does **not** scale with position
size. Its weight in a result therefore depends entirely on the assumed book size, which is a
parameter of the backtest rather than a property of the strategy. At the engine's default book of
₹10,00,000 spread across a hundred names, one full rebalance costs 100 x ₹15.34 = ₹1,534, or 15
basis points of the book, before a single proportional charge applies. At ₹100 crore the same
rebalance costs 0.0015 bps.

That is real — a retail investor genuinely pays it — but it means any conclusion about a
high-name-count strategy is partly a statement about assumed capital. This script isolates the term
by re-running the standard factors under three depository regimes, so the reader can see how much
of the gross-to-net collapse is proportional cost and how much is a fixed fee against a small book.

Usage:
    python scripts/dp_sensitivity.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402
from scripts.run_positive_control import FACTORS, load_strategy  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import summarise  # noqa: E402


def main() -> int:
    cfg = load_config()
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")

    # "none" is not a policy anyone should trade on. It is here as an ablation: it shows what the
    # net figure would be if the depository charge did not exist, which is the only way to read
    # how much of the drag the other two modes are carrying.
    modes = {
        "retail 15.34": cfg.costs,
        "research 3.50": replace(cfg.costs, dp_mode="research"),
        "none (ablation)": replace(
            cfg.costs, dp_charge_by_mode={**cfg.costs.dp_charge_by_mode, cfg.costs.dp_mode: 0.0}
        ),
    }
    engines = {
        label: BacktestEngine(panel=panel, calendar=calendar, universe=universe,
                              cost_model=CostModel(costs))
        for label, costs in modes.items()
    }

    print(f"\nBook size {1_000_000:,} rupees. Sharpe over the development window.\n")
    print(f"{'strategy':<30} {'gross':>8} " + " ".join(f"{label:>16}" for label in modes))
    for name, _family in FACTORS:
        cells: list[float] = []
        gross = 0.0
        for engine in engines.values():
            result = engine.run(load_strategy(name)(), cfg.dates.dev_start, cfg.dates.dev_end)
            gross = summarise(result.gross_returns)["sharpe_ratio"]
            cells.append(summarise(result.returns)["sharpe_ratio"])
        print(f"{name:<30} {gross:>8.4f} " + " ".join(f"{c:>16.4f}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
