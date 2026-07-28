"""Large-cap quality tilt for the Indian market.

Quality characteristics such as stable trailing returns per unit of volatility are best
measured on a set of genuinely large, liquid names, since ratio-based metrics get noisy for
thinly traded stocks. This module restricts trading to the well-known large-cap names that make
up the current headline index, so the ranking is not distorted by illiquid tickers.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

# The large, liquid names that make up the current NIFTY 100 headline list. Restricting the
# book to this set keeps the quality ranking away from thinly traded, noisy small caps.
_UNIVERSE: tuple[str, ...] = (
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND", "ADANIENT", "TATAMOTORS",
    "POWERGRID", "NTPC", "HCLTECH",
)
_LOOKBACK = 60
_MAX_NAMES = 10


def _quality_score(frame: pl.DataFrame) -> pl.DataFrame:
    """Rank each symbol by trailing return divided by trailing volatility."""
    ordered = frame.sort(["symbol", "session_date"])
    returns = ordered.with_columns(pl.col("adj_close").pct_change().over("symbol").alias("ret"))
    stats = returns.group_by("symbol").agg(
        trailing_return=(pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0),
        volatility=pl.col("ret").std(),
    )
    return stats.with_columns((pl.col("trailing_return") / pl.col("volatility")).alias("score"))


class LargeCapQualityTilt(Strategy):
    """Ranks a fixed list of well-known large caps by trailing return per unit of volatility."""

    rationale = (
        "Quality and stability characteristics are meaningful only against genuinely large, "
        "liquid names, since thin trading distorts return and volatility estimates for smaller "
        "stocks. Restricting the universe to the well-established large-cap names keeps the "
        "ranking clean and avoids chasing noise in illiquid tickers."
    )

    def generate(self, view: MarketView) -> Signal:
        frame = view.history(lookback=_LOOKBACK).filter(pl.col("symbol").is_in(_UNIVERSE))
        if frame.is_empty():
            return Signal(information_available_at=view.as_of, weights={})
        scored = _quality_score(frame).drop_nulls("score")
        ranked = scored.sort("score", descending=True).head(_MAX_NAMES)
        names = ranked["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        weight = 1.0 / len(names)
        return Signal(information_available_at=view.as_of, weights={s: weight for s in names})
