from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are likely to be more reliable signals of future "
        "strength. A stock with high volume on a price breakout is likely to continue moving in "
        "that direction as institutional and retail traders follow the trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].to_list()]
            volumes = [int(v) for v in history["volume"].to_list()]

            # Identify the breakout day
            breakout_day = None
            for i in range(1, len(prices)):
                if (
                    prices[i] > max(prices[:i]) and
                    volumes[i] > 1.5 * max(volumes[:i])
                ):
                    breakout_day = i
                    break

            # Check if a valid breakout was found
            if breakout_day is not None:
                # Confirm the move with subsequent volume on the breakout day
                for j in range(breakout_day + 1, len(prices)):
                    if volumes[j] > 0.9 * volumes[breakout_day]:
                        breakout_symbols.append(symbol)
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