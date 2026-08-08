from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to continue. "
        "By identifying strong volume surges following a price move, we can anticipate further momentum."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            close_prices = [float(v) for v in history["close"][symbol].drop_nulls().to_list()]
            volumes = [float(v) for v in history["volume"][symbol].drop_nulls().to_list()]

            if len(close_prices) < self._window + 1 or len(volumes) < self._window:
                continue

            price_change = close_prices[-1] - close_prices[0]
            volume_change = volumes[-1] / volumes[0]

            if price_change > 0 and volume_change >= self._threshold:
                symbol_data[symbol] = (price_change, volume_change)

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbol_data)
        selected_symbols = [s for s in symbol_data.keys()]
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest