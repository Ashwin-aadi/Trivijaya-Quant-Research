from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market momentum. "
        "By identifying symbols with both significant price movement and volume increase, we aim to capitalize on potential sustained trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        moves = {}
        for symbol in symbols:
            recent_closes = history[symbol].to_list()
            price_move = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

            if len(recent_closes) < self._window or price_move <= 0.05:
                continue

            volume_history = view.history(lookback=self._window)
            recent_volumes = [float(v) for v in volume_history[symbol]["volume"].to_list()]
            volume_change = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]

            if volume_change > 0.2:
                moves[symbol] = price_move

        if not moves:
            return Signal(information_available_at=stamp, weights={})

        sorted_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_moves[:5]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest