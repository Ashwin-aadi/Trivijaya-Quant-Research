from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are identified by observing a significant change in "
        "closing prices over a short period. High volume on the day of the move confirms its strength."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the price change over the window and the volume on the last day
            price_change = (values[-1] - values[0]) / abs(values[0])
            volume = view.history(symbol=symbol, lookback=1)["volume"].sum()
            if volume > 0:
                if price_change >= 0.05:  # Buy signal
                    picks.append(symbol)
                elif price_change <= -0.05:  # Sell signal
                    picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
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