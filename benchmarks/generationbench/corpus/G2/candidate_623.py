from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "By combining volume anomalies with recent price momentum, we aim to identify "
        "potentially overbought or oversold stocks that could reverse in the near future."
    )

    def __init__(self, window1: int = 20, window2: int = 5) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window1 + self._window2)

        if history.is_empty() or history.height < self._window1 + self._window2:
            return Signal(information_available_at=stamp, weights={})

        volumes = [float(v) for v in history["volume"].to_list()]
        closes = view.closes(lookback=self._window2)
        close_values = {symbol: float(close) for symbol, close in closes.to_dict().items()}

        momentum_symbols = []
        for symbol in view.symbols:
            if symbol not in close_values or len(volumes) < self._window1 + self._window2:
                continue
            recent_closes = [close_values[symbol]]
            if len(recent_closes) < self._window2:
                continue
            if (recent_closes[-1] - recent_closes[0]) / recent_closes[0] > 0.05:
                momentum_symbols.append(symbol)

        volume_symbols = []
        for symbol in view.symbols:
            if len(volumes) < self._window1 + self._window2 or volumes[-1] == 0:
                continue
            if (
                max(volumes[self._window1 :]) / volumes[-1]
                > (max(volumes[:self._window1]) / volumes[-1])
                * 1.5
            ):
                volume_symbols.append(symbol)

        potential_symbols = set(momentum_symbols).intersection(set(volume_symbols))
        if not potential_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(potential_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in potential_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest