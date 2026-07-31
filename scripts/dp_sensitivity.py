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

#: ₹10 lakh, ₹1 crore, ₹10 crore. A proportional cost is identical at all three; the depository
#: charge should fall roughly a hundredfold across them, and if it does not, the model is wrong.
BOOKS: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000)

#: The two factors where the depository charge is most and least implicated, plus the market
#: reference. Running the scale sweep over all eleven would triple an already slow script for no
#: additional information — the effect is a property of the charge, not of the strategy.
SCALE_FACTORS: tuple[tuple[str, str], ...] = (
    ("inverse_volatility_weighted", "many names, tiny daily adjustments - DP-dominated"),
    ("momentum_skip_month", "moderate turnover - statutory-dominated"),
    ("equal_weight_universe", "market reference - almost no turnover"),
)


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

    print(f"\nPART 1 — depository regime, at a book of {BOOKS[0]:,} rupees. "
          "Sharpe over the development window.\n")
    print(f"{'strategy':<30} {'gross':>8} " + " ".join(f"{label:>16}" for label in modes))
    for name, _family in FACTORS:
        cells: list[float] = []
        gross = 0.0
        for engine in engines.values():
            result = engine.run(load_strategy(name)(), cfg.dates.dev_start, cfg.dates.dev_end)
            gross = summarise(result.gross_returns)["sharpe_ratio"]
            cells.append(summarise(result.returns)["sharpe_ratio"])
        print(f"{name:<30} {gross:>8.4f} " + " ".join(f"{c:>16.4f}" for c in cells))

    # PART 2 — the scale question. A proportional charge is invariant to book size; the depository
    # charge is not, because it is a flat fee per scrip. If the DP contribution does not fall
    # roughly in proportion to the book, the model is not doing what it claims.
    retail = engines["retail 15.34"]
    without_dp = engines["none (ablation)"]
    print("\nPART 2 — cost in basis points per session, by book size.\n")
    print(f"{'strategy':<30} {'book':>14} {'total bps/d':>12} {'DP bps/d':>10} "
          f"{'DP share':>9} {'Sharpe':>9}")
    for name, _family in SCALE_FACTORS:
        for book in BOOKS:
            with_dp = retail.run(load_strategy(name)(), cfg.dates.dev_start, cfg.dates.dev_end,
                                 initial_equity=book)
            no_dp = without_dp.run(load_strategy(name)(), cfg.dates.dev_start, cfg.dates.dev_end,
                                   initial_equity=book)
            n = len(with_dp.costs) or 1
            total_bps = sum(with_dp.costs) / n * 10_000
            # The residual after removing the depository charge is the proportional part, so the
            # difference is the DP contribution measured rather than recomputed from the rate.
            dp_bps = total_bps - sum(no_dp.costs) / len(no_dp.costs or [1]) * 10_000
            share = dp_bps / total_bps if total_bps else 0.0
            print(f"{name:<30} {book:>14,} {total_bps:>12.3f} {dp_bps:>10.3f} "
                  f"{share:>8.1%} {summarise(with_dp.returns)['sharpe_ratio']:>9.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
