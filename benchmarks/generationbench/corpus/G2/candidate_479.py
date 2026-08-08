from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: volume-based momentum and "
        "price strength. It aims to identify stocks that have both high volume and strong price "
        "action over a recent period, suggesting underlying demand and support for the stock."
    )

    def __init__(self, window_volume: int = 20, threshold_volume: float = 1.5, window_price: int = 10) -> None:
        self._window_volume = window_volume
        self._threshold_volume = threshold_volume
        self._window_price = window_price

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_volume + self._window_price)

        if history.height < self._window_volume + self._window_price:
            return Signal(information_available_at=stamp, weights={})

        volume_signal = self._volume_momentum_signal(history)
        price_signal = self._price_strength_signal(history)

        combined_signal = {s: (v * 0.5 + p * 0.5) for s, v in volume_signal.items() if s in price_signal}
        
        if not combined_signal:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(combined_signal)
        return Signal(
            information_available_at=stamp, 
            weights={s: w for s, w in combined_signal.items()}
        )


    def _volume_momentum_signal(self, history: pl.DataFrame) -> dict[str, float]:
        volumes = {symbol: [float(v) for v in history[symbol].to_list()] for symbol in view.symbols}
        momentum_volumes = {}

        for symbol, volume_history in volumes.items():
            if len(volume_history) < self._window_volume:
                continue
            last_volume = volume_history[-1]
            max_volume = max(volume_history)
            if last_volume >= max_volume * self._threshold_volume:
                momentum_volumes[symbol] = 0.5

        return momentum_volumes


    def _price_strength_signal(self, history: pl.DataFrame) -> dict[str, float]:
        prices = {symbol: [float(v) for v in history["adj_close"][history.columns.get_index_of(symbol)].to_list()] for symbol in view.symbols}
        strength_prices = {}

        for symbol, price_history in prices.items():
            if len(price_history) < self._window_price:
                continue
            last_price = price_history[-1]
            max_price = max(price_history)
            if last_price >= max_price * 0.95:  # Weak threshold to identify strong price action
                strength_prices[symbol] = 0.5

        return strength_prices


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest