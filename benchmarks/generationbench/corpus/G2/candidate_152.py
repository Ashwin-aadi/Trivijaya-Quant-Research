from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate that a significant number of traders "
        "are committing to a particular direction. This can lead to sustained price movement and "
        "potential profit opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volatility_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].to_list()) < self._window:
                continue

            # Calculate daily returns
            returns = [float(v2 / v1 - 1) for v1, v2 in zip(history[symbol][:-1], history[symbol][1:])]

            # Calculate the volume on each day
            volumes = history["volume"].to_list()

            # Check if the return is positive and the volume increased or negative and the volume decreased
            directional_moves = [(r > 0) == (v2 > v1) for r, v1, v2 in zip(returns[:-1], volumes[:-1], volumes[1:])]

            if sum(directional_moves) >= 0.7 * len(directional_moves):
                high_volatility_symbols.append(symbol)

        if not high_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volatility_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in high_volatility_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest