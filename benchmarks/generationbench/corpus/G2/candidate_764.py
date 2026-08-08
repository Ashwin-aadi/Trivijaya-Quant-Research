from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume moves are often indicative of significant market interest or news events. "
        "If such a move is accompanied by continued high volume in subsequent sessions, it may "
        "signal strong conviction and the potential for sustained price movement."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.is_empty() or history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol]["adj_close"].drop_nulls().to_list()]
            volume_values = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()]

            # Find the first day with a significant move
            for i in range(len(close_values) - 1, self._window, -1):
                if (close_values[i] / close_values[i - 1] - 1.0) > 0.05 or \
                   (close_values[i] / close_values[i - 1] - 1.0) < -0.05:
                    break

            # Check if volume is high in the subsequent days
            for i in range(i + 1, min(len(close_values), self._window + 2)):
                if (close_values[i] / close_values[i - 1] - 1.0) < 0.02 and \
                   volume_values[i] > 1.5 * volume_values[i - 1]:
                    breakout_symbols.add(symbol)
                    break

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest