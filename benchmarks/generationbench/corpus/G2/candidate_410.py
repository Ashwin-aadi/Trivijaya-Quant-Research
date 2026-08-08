from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the tendency of assets to continue trending "
        "in their recent direction after a period of heightened volatility. During such periods, "
        "trading in the direction of the trend can yield positive returns."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        daily_returns = (history.select(pl.col("adj_close").to_list()) / history.shift(1).select(
            pl.col("adj_close").to_list()
        ) - 1.0)

        # Calculate volatility over the window period
        volatility = daily_returns.std().item()

        # Filter out symbols with insufficient data or very low volatility
        if volatility < self._threshold:
            return Signal(information_available_at=stamp, weights={})

        recent_trend = (daily_returns.tail(self._window).sum() > 0.0).all()
        if not recent_trend:
            return Signal(information_available_at=stamp, weights={})

        # Identify symbols with strong trends
        strong_trends: list[str] = []
        for symbol in view.symbols:
            trend_strength = daily_returns[symbol].sum().item() / volatility
            if trend_strength > 1.0:
                strong_trends.append(symbol)

        weight = 1.0 / len(strong_trends)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in strong_trends}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest