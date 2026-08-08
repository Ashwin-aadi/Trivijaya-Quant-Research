from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that large volumes are often associated with "
        "significant price changes. These changes can be profit opportunities if they are timely "
        "identified and acted upon."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            volumes = view.history()[view.history()["symbol"] == symbol]["volume"].to_list()

            if len(recent_closes) < self._window or len(volumes) < self._window:
                continue

            last_close = recent_closes[-1]
            last_volume = volumes[-1]

            # Check for a significant price change
            price_change = (last_close - recent_closes[0]) / recent_closes[0] * 100.0
            if abs(price_change) > 2:  # Threshold for significant price change

                # Ensure this is volume-confirmed
                average_volume = sum(volumes) / len(volumes)
                if last_volume > average_volume * 1.5:
                    signals[symbol] = 1.0  # Assign full weight to the symbol

        return Signal(information_available_at=stamp, weights={s: w for s, w in signals.items() if w > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest