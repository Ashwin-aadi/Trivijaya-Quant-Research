from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following is a strategy that bets on the continuation of trends "
        "but scales these bets by an estimate of recent volatility. High volatility periods tend to "
        "follow low volatility periods and vice versa, leading to mean reversion in returns. By scaling "
        "trends with this factor, we aim to capture more significant moves while mitigating risk during volatile times."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        rets = (
            (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0).alias("ret")
            if not closes.is_empty()
            else pl.DataFrame({"ret": []})
        )

        # Filter out the first row as it doesn't have a prior value
        rets = rets.filter(pl.col("session_date") > view.as_of.date())

        # Calculate volatility over the last `vol_window` days for each symbol
        vol = (
            history.group_by("symbol")
            .agg(
                (pl.col("ret").abs().mean().alias("volatility"))
                if not rets.is_empty()
                else pl.DataFrame({"volatility": []})
            )
            .with_columns(pl.col("volatility").shift(1).fill_null(0.0).alias("prev_volatility"))
        )

        # Create a weight for each symbol
        weights = (
            history.group_by("symbol")
            .agg(
                (pl.col("ret").sum() / pl.lit(self._window)).alias("trend")
                if not rets.is_empty()
                else pl.DataFrame({"trend": []})
            )
            .with_columns(
                ((pl.col("trend") * 2.0) / (1.0 + vol["prev_volatility"])).alias("weighted_trend")
            )
        )

        # Select the top N symbols based on weighted trend
        picks = (
            weights.sort("weighted_trend", descending=True)
            .select(["symbol"])
            .to_series()
            .to_list()[: self._vol_window]
        )

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest