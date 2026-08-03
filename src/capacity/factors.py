"""Factor signals built from price and volume alone, because that is all this repository holds.

**What is missing and why, stated here rather than in a footnote.** The charter's factor zoo lists
momentum, short-term reversal, **value**, **quality** and low-volatility. Value and quality require
company fundamentals — book value, earnings, accruals, return on equity — which this repository does
not hold and cannot acquire under the PI's ruling of 2026-08-03 fixing the market-data budget at ₹0
with no account creation. **They are therefore not built.** The zoo below is a price-and-volume zoo,
and every comparison against published factor results must be read knowing that the two most
fundamentals-dependent families are absent. This weakens the cross-market decay comparison in RQ2
and is reported as a limitation rather than absorbed silently.

Every signal is point-in-time by construction: each uses a trailing window ending strictly before
the session it labels. That is enforced here by shifting before rolling, not by convention, for the
same reason :mod:`src.backtest` enforces it structurally — a factor that peeks is indistinguishable
from a good factor until someone checks.
"""

from __future__ import annotations

import polars as pl

from src.common.exceptions import DataIntegrityError

#: Sessions in a year, a month and a half-year of trading, used for the trailing windows below.
YEAR = 252
MONTH = 21
HALF_YEAR = 126

#: Every factor this module can build. Value and quality are deliberately absent; see the module
#: docstring. Kept as a tuple so a caller cannot silently receive a shorter zoo than it expected.
FACTOR_NAMES = (
    "momentum_12_1",
    "reversal_1m",
    "low_volatility",
    "illiquidity",
    "liquidity_size_proxy",
)


def daily_universe(universe: pl.DataFrame, sessions: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Expand quarterly rebalance snapshots into per-session membership.

    Membership on session ``t`` is the most recent rebalance at or before ``t``. A session earlier
    than the first rebalance has no membership at all rather than borrowing the first one, which
    would apply a universe selected on later information to earlier dates.
    """
    if "rebalance_date" not in universe.columns or "symbol" not in universe.columns:
        raise DataIntegrityError("universe frame needs rebalance_date and symbol columns")
    calendar = pl.DataFrame({"session_date": sessions}).sort("session_date")

    # Two steps, and the second is the one that is easy to get wrong. An as-of join matches ONE
    # right row per left row, so joining the universe directly would return a single arbitrary
    # symbol per session rather than that session's whole membership. So: map each session to its
    # applicable rebalance date first, against the DISTINCT rebalance dates, then expand.
    rebalances = universe.select("rebalance_date").unique().sort("rebalance_date")
    applicable = calendar.join_asof(
        rebalances, left_on="session_date", right_on="rebalance_date", strategy="backward"
    ).drop_nulls("rebalance_date")

    expanded = applicable.join(universe, on="rebalance_date", how="inner")
    if expanded.height <= calendar.height:
        raise DataIntegrityError(
            f"daily universe expanded to {expanded.height} rows from {calendar.height} sessions; "
            "a correct expansion carries many symbols per session, so this is the as-of-join "
            "collapse described above rather than a genuinely tiny universe"
        )
    return expanded.select(["session_date", "symbol", "rebalance_date"])


def build_factors(panel: pl.DataFrame) -> pl.DataFrame:
    """Return a long frame of ``session_date, symbol, factor, score`` over the whole panel.

    Scores are raw, not standardised. Ranking happens at portfolio construction, so that a factor's
    cross-sectional distribution stays inspectable here rather than being normalised away before
    anyone has looked at it.
    """
    required = {"session_date", "symbol", "adj_close", "turnover_inr"}
    missing = required - set(panel.columns)
    if missing:
        raise DataIntegrityError(f"panel is missing columns {sorted(missing)}")

    ordered = panel.sort(["symbol", "session_date"]).with_columns(
        ret=(pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0)
    )
    scored = ordered.with_columns(
        # Twelve-month return skipping the most recent month, the standard construction: the
        # skipped month is where short-horizon reversal lives, and including it mixes two effects
        # with opposite signs into one signal.
        momentum_12_1=(
            pl.col("adj_close").shift(MONTH).over("symbol")
            / pl.col("adj_close").shift(YEAR).over("symbol")
            - 1.0
        ),
        # Negated so that a high score is the side the factor goes long: last month's losers.
        reversal_1m=-(
            pl.col("adj_close") / pl.col("adj_close").shift(MONTH).over("symbol") - 1.0
        ),
        # Negated likewise: high score means low realised volatility.
        low_volatility=-(
            pl.col("ret")
            .shift(1)
            .rolling_std(window_size=HALF_YEAR, min_samples=HALF_YEAR)
            .over("symbol")
        ),
        # Amihud averaged over the trailing half-year. High score means illiquid, which is the
        # side the illiquidity premium is claimed to pay.
        illiquidity=(
            pl.when(pl.col("turnover_inr") > 0)
            .then(pl.col("ret").abs() / pl.col("turnover_inr"))
            .otherwise(None)
            .shift(1)
            .rolling_mean(window_size=HALF_YEAR, min_samples=HALF_YEAR // 2)
            .over("symbol")
        ),
        # Trailing traded value, negated so a high score means a smaller name. **Named for what it
        # is.** It was called `size` until the PI ruled on 2026-08-03 that the name claimed a factor
        # the data does not support: market capitalisation needs a share count this repository does
        # not hold, and traded value is a liquidity measure that correlates with size rather than a
        # measure of size. The old name would have invited a reader to compare it against published
        # size-factor results, which would not be a like-for-like comparison.
        liquidity_size_proxy=-(
            pl.col("turnover_inr")
            .shift(1)
            .rolling_mean(window_size=HALF_YEAR, min_samples=HALF_YEAR)
            .over("symbol")
            .log1p()
        ),
    )
    return (
        scored.select(["session_date", "symbol", *FACTOR_NAMES])
        .unpivot(
            index=["session_date", "symbol"],
            variable_name="factor",
            value_name="score",
        )
        .drop_nulls("score")
    )


def long_short_weights(
    scores: pl.DataFrame,
    *,
    quantile: float = 0.2,
) -> pl.DataFrame:
    """Equal-weighted long-short book from the top and bottom ``quantile`` of each cross-section.

    Weights sum to +1 on the long side and -1 on the short side, so the book is rupee-neutral and a
    capacity figure computed from it scales linearly in deployed AUM. Cross-sections with too few
    names to form both legs are dropped rather than concentrated into a one-name portfolio.
    """
    if not 0 < quantile < 0.5:
        raise DataIntegrityError(f"quantile must lie in (0, 0.5), got {quantile}")
    ranked = scores.with_columns(
        rank=pl.col("score").rank("ordinal").over(["session_date", "factor"]),
        n=pl.len().over(["session_date", "factor"]),
    ).filter(pl.col("n") >= 10)  # below this a quintile is one or two names, not a portfolio
    cut = (pl.col("n") * quantile).ceil()
    legs = ranked.with_columns(
        side=pl.when(pl.col("rank") > pl.col("n") - cut)
        .then(pl.lit(1.0))
        .when(pl.col("rank") <= cut)
        .then(pl.lit(-1.0))
        .otherwise(pl.lit(0.0))
    ).filter(pl.col("side") != 0.0)
    return legs.with_columns(
        weight=pl.col("side") / pl.len().over(["session_date", "factor", "side"])
    ).select(["session_date", "factor", "symbol", "weight"])
