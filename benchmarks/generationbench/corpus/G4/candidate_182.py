from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages the tendency for markets to revert to recent volatility levels "
        "after periods of high or low volatility. By dynamically scaling positions based on current "
        "volatility, it aims to capture trends more effectively while managing risk in turbulent times."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (history.open / history.close.shift(1) - 1.0).alias("return")
        returns_df = history.with_columns(returns)

        # Calculate realized volatility over the window period
        vol_col = pl.col("return").abs().mean() * (252 ** 0.5)
        vol_df = returns_df.select(vol_col.alias("volatility"))

        # Compute moving average of historical volatility
        vol_windowed = vol_df.sort("session_date").tail(self._window).select(
            pl.col("volatility").mean().alias("moving_vol")
        )
        latest_moving_vol = float(vol_windowed[0]["moving_vol"])

        # Scale positions based on the current volatility compared to its historical average
        total_weight = 1.0
        weight_per_symbol = latest_moving_vol / latest_moving_vol.mean() * (total_weight / len(view.symbols))
        
        if weight_per_symbol == 0:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: weight_per_symbol for symbol in view.symbols}
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest