from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of significant market "
        "sentiment shifts. By identifying symbols that show a strong and sustained price movement "
        "across multiple sessions with concurrent volume increases, we can capture potential "
        "opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_symbols: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            recent_closes = [float(v) for v in symbol_history["adj_close"].to_list()]
            if len(recent_closes) < self._window + 1:
                continue

            price_changes = [
                (recent_closes[i] - recent_closes[i - 1]) / recent_closes[i - 1]
                for i in range(1, len(recent_closes))
            ]
            volume_changes = [float(v) for v in symbol_history["volume"].to_list()]
            if all(
                abs(change) > self._threshold and vol_change > 0
                for change, vol_change in zip(price_changes, volume_changes)
            ):
                volume_confirmed_symbols.append(symbol)

        if not volume_confirmed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_confirmed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in volume_confirmed_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date), "Expected session_date to be of type date"
    return newest