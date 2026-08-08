from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "sustained price action. By identifying such moves, we aim to capitalize on the momentum."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or history.width < 1 + len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        directional_moves = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = history[symbol].drop_nulls().to_list()
            volumes = history[symbol + "_volume"].drop_nulls().to_list()

            if len(closes) < self._window or len(volumes) < self._window:
                continue

            for i in range(1, len(closes)):
                prev_close = closes[i - 1]
                current_close = closes[i]
                vol_ratio = volumes[i] / volumes[i - 1]

                if (prev_close < current_close and vol_ratio >= self._min_volume_ratio) or \
                   (prev_close > current_close and vol_ratio <= 1 / self._min_volume_ratio):
                    directional_moves.append(symbol)
                    break

        directional_moves = list(set(directional_moves))[:5]
        if not directional_moves:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(directional_moves)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in directional_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest