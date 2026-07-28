"""Split the buy-and-hold tracking error into the two choices that cause it.

At Checkpoint 1.2 the engine's buy-and-hold differed from the NIFTY total-return series, and the
tracking error was accepted as engine evidence with the decomposition deferred to a Phase 1.4
prerequisite. This is that decomposition.

Two deliberate departures from the index sit between our buy-and-hold and NIFTY, and reporting a
single tracking-error number conflates them:

1. **Weighting.** We hold the universe equally weighted; the index is capitalisation weighted. On
   any day the two differ, and the difference is a real, intended exposure choice - equal weighting
   is a small-cap tilt against the index.
2. **Membership.** Our universe is a liquidity-screened rules-based reconstruction, not the actual
   published constituent list, because point-in-time membership was not obtainable from free
   sources. Any name we hold that the index did not, or the reverse, contributes error that is a
   *data limitation* rather than a design choice.

These have opposite implications. The first is a decision the PI approved and would not want
removed; the second bounds how much of the engine's disagreement with the index is our data being
imperfect. The point of separating them is to keep the second from hiding inside the first.

Usage:
    python scripts/decompose_tracking_error.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.eval.metrics import annualised_volatility, tracking_error  # noqa: E402

_log = get_logger(__name__)


def daily_returns(frame: pl.DataFrame, value_column: str) -> list[float]:
    """Simple returns of a single-column daily series, oldest first."""
    ordered = frame.sort("session_date")
    values = ordered[value_column].to_list()
    return [
        float(b) / float(a) - 1.0
        for a, b in zip(values, values[1:], strict=False)
        if a not in (None, 0) and b is not None
    ]


def expand_membership(universe: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Turn the rebalance-dated constituent table into one row per session per member.

    Mirrors `BacktestEngine._universe_on`: on any session the members are those from the most
    recent rebalance at or before it. Re-deriving membership by a different rule here would make
    the decomposition describe a universe the engine never traded.
    """
    sessions = panel.select("session_date").unique().sort("session_date")
    rebalances = universe.select("rebalance_date").unique().sort("rebalance_date")
    # A backward as-of join assigns each session the latest rebalance not after it.
    mapped = sessions.join_asof(
        rebalances.with_columns(pl.col("rebalance_date").alias("effective")),
        left_on="session_date",
        right_on="rebalance_date",
        strategy="backward",
    ).drop_nulls("effective")
    return (
        mapped.join(universe, left_on="effective", right_on="rebalance_date", how="inner")
        .select(["symbol", "session_date"])
        .unique()
    )


def equal_weight_returns(panel: pl.DataFrame, symbols_by_date: pl.DataFrame) -> pl.DataFrame:
    """Daily return of an equally weighted basket of whatever was in the universe that day."""
    per_name = (
        panel.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0).alias("r")
        )
        .drop_nulls("r")
    )
    eligible = per_name.join(symbols_by_date, on=["symbol", "session_date"], how="inner")
    return eligible.group_by("session_date").agg(pl.col("r").mean().alias("equal_weight")).sort(
        "session_date"
    )


def cap_weight_proxy(panel: pl.DataFrame, symbols_by_date: pl.DataFrame) -> pl.DataFrame:
    """Turnover-weighted basket over the same names, standing in for capitalisation weighting.

    Free sources give no point-in-time share count, so true capitalisation weights are not
    available. Traded value is the closest observable proxy and is stated as an approximation here
    rather than presented as the index's own weighting - the residual it leaves is therefore an
    upper bound on the weighting effect, not a measurement of it.
    """
    per_name = (
        panel.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0).alias("r"),
            (pl.col("close") * pl.col("volume")).alias("traded_value"),
        )
        .drop_nulls("r")
    )
    eligible = per_name.join(symbols_by_date, on=["symbol", "session_date"], how="inner")
    return (
        eligible.group_by("session_date")
        .agg(
            (
                (pl.col("r") * pl.col("traded_value")).sum() / pl.col("traded_value").sum()
            ).alias("cap_proxy")
        )
        .sort("session_date")
    )


def main() -> int:
    cfg = load_config()
    processed = cfg.paths.data_processed
    panel = pl.read_parquet(processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(processed / "universe.parquet")

    start, end = cfg.dates.dev_start, cfg.dates.dev_end
    panel = panel.filter(
        (pl.col("session_date") >= start) & (pl.col("session_date") <= end)
    )

    index_path = cfg.paths.data_raw / "calendar_cnx100.parquet"
    index_frame = pl.read_parquet(index_path).filter(
        (pl.col("session_date") >= start) & (pl.col("session_date") <= end)
    )
    level_column = next(
        (c for c in ("close", "adj_close", "level", "Close") if c in index_frame.columns), None
    )
    if level_column is None:
        _log.error("no index level column in %s; columns are %s", index_path, index_frame.columns)
        return 1

    members = expand_membership(universe, panel)
    equal = equal_weight_returns(panel, members)
    proxy = cap_weight_proxy(panel, members)

    joined = equal.join(proxy, on="session_date", how="inner").join(
        index_frame.select(["session_date", pl.col(level_column).alias("index_level")]),
        on="session_date", how="inner",
    ).sort("session_date")

    index_returns = daily_returns(
        joined.select(["session_date", "index_level"]), "index_level"
    )
    equal_returns = joined["equal_weight"].to_list()[1:]
    proxy_returns = joined["cap_proxy"].to_list()[1:]
    n = min(len(index_returns), len(equal_returns), len(proxy_returns))
    index_returns, equal_returns = index_returns[:n], equal_returns[:n]
    proxy_returns = proxy_returns[:n]

    total = tracking_error(equal_returns, index_returns)
    weighting = tracking_error(equal_returns, proxy_returns)
    membership = tracking_error(proxy_returns, index_returns)

    report = {
        "sessions": n,
        "window": [str(start), str(end)],
        "total_tracking_error": total,
        "weighting_component": weighting,
        "membership_and_residual_component": membership,
        "equal_weight_volatility": annualised_volatility(equal_returns),
        "index_volatility": annualised_volatility(index_returns),
    }
    (processed / "tracking_error_decomposition.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    proxy_failed = weighting > total or membership > total
    report["proxy_usable"] = not proxy_failed

    print(f"sessions: {n}")
    print(f"total tracking error (equal weight vs index):        {total:.4f}")
    print(f"  via traded-value proxy, weighting leg:             {weighting:.4f}")
    print(f"  via traded-value proxy, membership leg:            {membership:.4f}")

    if proxy_failed:
        print(
            "\nTHE DECOMPOSITION FAILED, AND THAT IS THE RESULT.\n"
            "Both legs exceed the total they are meant to partition. The traded-value proxy sits "
            "further from the index than equal weighting does, so it is not standing in for "
            "capitalisation weighting at all - turnover weighting concentrates on whatever traded "
            "heavily that day, which is a different portfolio, not a closer one.\n"
            "The two legs cannot therefore be read as a weighting share and a membership share. "
            "Only the total is reported. Separating them needs point-in-time free-float market "
            "capitalisation, which was not obtainable from free sources, and no substitute here "
            "should be tuned until the numbers look additive."
        )
    else:
        print("\nComponents are magnitudes, not an additive split: tracking errors are standard "
              "deviations of correlated differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
