from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After an initial breakout, if the security's price continues to move in the "
        "direction of the breakout for a certain period, it suggests strong momentum and "
        "potentially favorable continuation. This strategy identifies such securities."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + self._continuation_window)
        if closes.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_window:
                continue

            breakout_price = max(values[-self._window:])
            last_price = values[-1]

            if last_price > breakout_price and all(
                values[i] > values[i - 1]
                for i in range(-self._continuation_window, 0)
            ):
                picks.append(symbol)

        picks = picks[:5]
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