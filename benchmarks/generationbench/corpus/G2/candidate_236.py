from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Strong directional moves in price are often accompanied by increased volume. "
        "Such volume-confirmed moves could indicate a stronger momentum or trend in the market. "
        "By identifying these moves, we can capture potentially significant gains."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_signals: dict[str, float] = {}

        for symbol in symbols:
            prices = history[symbol].to_list()
            volumes = history["volume"][symbol].to_list()

            price_change = (prices[-1] - prices[0]) / prices[0]
            volume_change = (volumes[-1] - volumes[0]) / volumes[0]

            if abs(price_change) > 0.05 and abs(volume_change) > 0.1:
                volume_confirmed_signals[symbol] = price_change

        if not volume_confirmed_signals:
            return Signal(information_available_at=stamp, weights={})

        sorted_signals = sorted(volume_confirmed_signals.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbol = sorted_signals[0][0]
        weight = 1.0 / len(sorted_signals)

        return Signal(
            information_available_at=stamp,
            weights={top_symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest