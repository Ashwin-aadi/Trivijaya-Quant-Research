from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "sustained price movements. This strategy identifies such moves by combining direction with"
        "volume data."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volume_confirmed_moves = []

        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol)
            if row.is_empty():
                continue

            closes = row["adj_close"].to_list()
            volumes = row["volume"].to_list()

            # Calculate the percentage change in close price and volume
            close_changes = [0.0] + [(c / p - 1.0) for c, p in zip(closes[1:], closes[:-1])]
            volume_changes = [0.0] + [(v / v_prev - 1.0) for v, v_prev in zip(volumes[1:], volumes[:-1])]

            # Find days where both close and volume changes are significant
            significant_moves = [
                (close_changes[i] > 0.02) & (abs(volume_changes[i]) > 0.5)
                for i in range(len(close_changes))
            ]

            if any(significant_moves):
                volume_confirmed_moves.append(symbol)

        if not volume_confirmed_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_confirmed_moves)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume_confirmed_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_pydatetime().date()
    assert isinstance(newest, date)
    return newest