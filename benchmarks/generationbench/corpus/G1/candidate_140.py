from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are more likely to continue than those that "
        "occur without strong volume support. This strategy seeks to capture such moves."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_price = [float(v) for v in history[symbol]["open"].drop_nulls().to_list()]
            close_price = [float(v) for v in history[symbol]["close"].drop_nulls().to_list()]
            volume = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()]

            if len(open_price) < self._window or len(close_price) < self._window or len(volume) < self._window:
                continue

            # Calculate daily returns
            returns = [(close / open - 1.0) for close, open in zip(close_price[1:], open_price[:-1])]
            # Filter out days with no return (e.g., equal opens and closes)
            returns = [r for r in returns if r != 0]

            # Calculate volume-weighted returns
            vwr = sum(returns[i] * volume[i + 1] for i in range(len(returns))) / sum(volume[1:])

            if vwr > 0.0:
                picks[symbol] = abs(vwr)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Normalize the picks
        total_weight = sum(picks.values())
        normalized_weights = {k: v / total_weight for k, v in picks.items()}
        return Signal(
            information_available_at=stamp,
            weights=normalized_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest