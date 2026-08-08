from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend following involves capturing gains in the direction of a trend. "
        "Volatility scaling adjusts position sizes to maintain consistent risk exposure regardless "
        "of market conditions."
    )

    def __init__(self, window: int = 20, volatility_multiplier: float = 1.5) -> None:
        self._window = window
        self._volatility_multiplier = volatility_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or "adj_close" not in history.columns:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the rolling returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .select(["symbol", "session_date", "return"])
        )

        # Calculate the rolling standard deviation of returns
        history = (
            history.with_columns(
                pl.col("return").rolling_std(window=self._window, center=False).alias("volatility")
            )
        )

        # Filter for symbols with non-zero volatility to avoid division by zero
        filtered_history = history.filter(pl.col("volatility") > 0.0)

        # Calculate the scaled returns and ranks
        scaled_returns = (
            filtered_history.with_columns(
                (pl.col("return") * self._volatility_multiplier / pl.col("volatility")).alias("scaled_return")
            )
            .group_by(["symbol"])
            .agg(pl.col("scaled_return").mean().alias("average_scaled_return"))
            .sort("average_scaled_return", descending=True)
        )

        top_symbols = scaled_returns.head(self._window)["symbol"].to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest