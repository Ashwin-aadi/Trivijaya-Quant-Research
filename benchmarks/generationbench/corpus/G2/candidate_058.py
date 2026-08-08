from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTradingStrategy(Strategy):
    rationale = (
        "Certain months of the year have historically seen higher trading volumes and "
        "prices for stocks in India due to seasonal effects. By identifying these periods, we "
        "can exploit such trends to generate positive returns."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean close price of the last month
            recent_closes = values[-30:]
            recent_mean = sum(recent_closes) / len(recent_closes)
            current_close = values[-1]

            # Check if the current close is significantly higher than the mean
            if current_close > 1.2 * recent_mean:
                picks.append(symbol)

        picks = picks[: self._top_n]
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