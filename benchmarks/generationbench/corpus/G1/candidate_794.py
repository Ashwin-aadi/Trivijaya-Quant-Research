from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify and follow the prevailing trend "
        "in the market while scaling positions based on recent volatility. High volatility periods"
        " suggest increased uncertainty and risk, prompting smaller position sizes."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window)

        if history.height < self._window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        vol_series = (closes / closes.shift(1) - 1.0).abs().rolling_sum(window=self._vol_window)
        recent_volatility = float(vol_series.max())

        # Calculate the trend direction
        trend_direction = pl.col("close").shift(-self._window).rank(method="dense", descending=True)
        top_trending_symbols = [symbol for symbol in view.symbols if trend_direction[symbol] == 1]
        
        # Adjust weights based on volatility
        weight = recent_volatility / len(top_trending_symbols) if top_trending_symbols else 0.0
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_trending_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest