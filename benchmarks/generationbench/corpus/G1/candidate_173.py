from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "If a security breaks out of its recent range and then closes within that range on the "
        "next session, it signals continued momentum in the breakout direction. This can be used "
        "to identify potentially strong trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol]) < self._window + 2:
                continue

            close_lag1 = float(history[symbol].to_list()[-2])
            close_current = float(history[symbol].to_list()[-1])

            open_ = float(history[symbol][0])
            high = max(float(v) for v in history[symbol][1:self._window + 1])
            low = min(float(v) for v in history[symbol][1:self._window + 1])

            if close_current <= high and close_current >= low:
                picks.append(symbol)

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