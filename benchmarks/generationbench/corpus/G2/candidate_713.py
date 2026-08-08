from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continued price movement in the same direction. "
        "By identifying stocks that have recently broken out and continue to rise or fall, we can "
        "potentially profit from this momentum."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            session_dates = history["session_date"].to_list()
            opens = [float(v) for v in history[symbol][0].drop_nulls().to_list()]
            closes = [float(v) for v in history[symbol][-1].drop_nulls().to_list()]

            if len(opens) < self._window + 2:
                continue

            breakout_date = session_dates[-self._window - 1]
            open_price = opens[-(self._window + 1)]
            close_price = closes[-self._window]

            if (open_price > close_price and
                    history[symbol][0][session_dates.index(breakout_date):].max() >= close_price) or \
               (open_price < close_price and
                    history[symbol][0][session_dates.index(breakout_date):].min() <= close_price):
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))[:self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest