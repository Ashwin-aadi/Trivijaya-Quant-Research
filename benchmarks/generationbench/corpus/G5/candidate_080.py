from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong momentum in a security. "
        "By identifying such moves, we can capitalize on the likely continuation of trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_low_diff = [float(v) - float(h) for h, v in zip(history[symbol]["low"].to_list(), history[symbol]["high"].to_list())]
            volume_change = [
                (float(v1) / float(v2)) - 1.0 if float(v2) != 0 else 0.0
                for v1, v2 in zip(history[symbol]["volume"].shift(-1).to_list(), history[symbol]["volume"].to_list())
            ]
            up_days = [diff > 0 and change >= 0.05 for diff, change in zip(high_low_diff, volume_change)]
            
            if any(up_days):
                picks.append(symbol)

        picks = picks[:2]  # Limit to top 2 symbols
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest