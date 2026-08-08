from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrend(Strategy):
    rationale = (
        "Volatility-scaled trend following involves entering trades based on the recent "
        "price movement and adjusting position sizes according to historical volatility. "
        "Higher volatility periods suggest greater uncertainty and potential for larger price "
        "moves, while lower volatility periods indicate a more stable market where trends are "
        "less likely."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            (history["close"] / history["close"].shift(1) - 1.0).abs().mean()
        ).item()

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_returns = [
            (float(closes[symbol].to_list()[-2] - closes[symbol].to_list()[0]) / closes[symbol].to_list()[0])
            for symbol in view.symbols
        ]
        
        top_gainers = [symbol for i, symbol in enumerate(view.symbols) if recent_returns[i] >= volatility * self._threshold]
        bottom_losers = [symbol for i, symbol in enumerate(view.symbols) if recent_returns[i] <= -volatility * self._threshold]

        weights: dict[str, float] = {}
        total_weight = 1.0
        if top_gainers:
            weight = total_weight / len(top_gainers)
            for symbol in top_gainers:
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest