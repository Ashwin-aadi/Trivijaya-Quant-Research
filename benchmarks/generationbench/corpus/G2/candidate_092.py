from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume breaks in a clear directional move can indicate significant market "
        "sentiment or news impact. Such moves often lead to continuation of the trend, "
        "offering profitable trading opportunities."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            close_prices = [float(v) for v in df["adj_close"].to_list()]
            volumes = [float(v) for v in df["volume"].to_list()]

            if len(close_prices) < self._window + 1:
                continue

            # Calculate directional move
            last_price = close_prices[-1]
            prev_price = close_prices[-2]
            price_move = (last_price - prev_price) / prev_price
            volume_change = volumes[-1] - volumes[-2]

            if abs(price_move) < 0.05 and abs(volume_change) < self._threshold * max(volumes):
                continue

            symbol_data[symbol] = {
                "price_move": price_move,
                "volume_change": volume_change
            }

        # Filter out symbols that do not meet the criteria
        filtered_symbols = [
            s for s in symbol_data.keys()
            if abs(symbol_data[s]["price_move"]) > 0.1 and abs(symbol_data[s]["volume_change"]) > self._threshold * max(volumes)
        ]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest