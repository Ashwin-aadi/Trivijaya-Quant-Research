from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following involves entering a position when the price "
        "crosses above its 20-day moving average. The scaling factor is based on the historical "
        "volatility of the asset, making it more aggressive during periods of high volatility."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        volatilities = {}
        for symbol in symbols:
            prices = history[symbol].to_list()
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, self._window + 1)]
            volatility = sum([abs(r) for r in returns]) / self._window
            volatilities[symbol] = volatility

        signals: dict[str, float] = {}
        for symbol in symbols:
            last_close = closes[symbol].to_list()[-1]
            moving_average = history["adj_close"][symbol].mean()
            if last_close > moving_average * (1 + self._scale_factor * volatilities.get(symbol, 0)):
                signals[symbol] = self._scale_factor

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_scale = sum(signals.values())
        weights = {symbol: signal / total_scale for symbol, signal in signals.items()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest