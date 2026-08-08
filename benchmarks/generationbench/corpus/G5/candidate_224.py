from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A directional move in price that is confirmed by increased volume suggests "
        "stronger market sentiment and is likely to continue. This strategy identifies such moves."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            recent_close = float(symbol_history["adj_close"].tail(1)[0])
            last_close = float(symbol_history["adj_close"][1])
            close_diff = (recent_close - last_close) / last_close
            if abs(close_diff) < 0.01:  # Ignore insignificant moves
                continue

            avg_volume = symbol_history.select(pl.col("volume").mean()).item()
            current_volume_sum = symbol_history.select(pl.col("volume").sum()).item()
            volume_ratio = current_volume_sum / (len(symbol_history) * avg_volume)
            if close_diff > 0 and volume_ratio > 1.2:
                volume_confirmed[symbol] = abs(close_diff)
            elif close_diff < 0 and volume_ratio > 1.5:
                volume_confirmed[symbol] = -abs(close_diff)

        if not volume_confirmed:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [
            symbol
            for _, symbol in sorted(volume_confirmed.items(), key=lambda item: abs(item[1]), reverse=True)
        ]
        top_symbol = sorted_symbols[0]
        weight = 1.0 / len(volume_confirmed)
        return Signal(
            information_available_at=stamp, weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest