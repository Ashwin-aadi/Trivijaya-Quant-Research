from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalBreakout(Strategy):
    rationale = (
        "Stock markets often exhibit seasonality effects, where certain months or seasons "
        "of the year have historically produced higher returns. By identifying stocks that "
        "break out of their recent range during these periods, we can capture potential "
        "profit opportunities."
    )

    def __init__(self, window: int = 20, breakout_threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = breakout_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify seasonal breakout candidates
        breakout_candidates = []
        for symbol in view.symbols:
            closes = [float(v) for v in history[symbol].to_list()]
            last_close = closes[-1]
            mean_price = sum(closes) / len(closes)
            if last_close > self._threshold * mean_price:
                breakout_candidates.append(symbol)

        # Filter by the month of the latest close
        today_month = stamp.month
        breakout_symbols = [s for s in breakout_candidates if history[symbol].tail(1)["session_date"].month == today_month]

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest