from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often signals of significant price action. "
        "A large volume increase alongside a price breakout can indicate strong momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < (self._window + 1):
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        recent_closes = [float(v) for v in latest_closes.drop_nulls().to_list()]
        recent_volume = history["volume"].drop_nulls().to_list()

        breakout_symbols: list[str] = []
        for i, (symbol, vol) in enumerate(zip(recent_closes, recent_volume)):
            if symbol not in history.columns:
                continue
            prev_close = float(history[symbol].shift(1).drop_nulls().tail(n=1)[0])
            if vol > max(recent_volume) and symbol != prev_close.symbol:
                breakout_symbols.append(symbol)

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