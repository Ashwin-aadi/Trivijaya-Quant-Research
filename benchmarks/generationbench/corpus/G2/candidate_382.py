from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality can be a powerful driver of stock returns. If certain months historically "
        "see higher returns for specific stocks, we can exploit this by overweighing those "
        "stocks during the favorable seasons."
    )

    def __init__(self, window: int = 20, favorable_months: list[int] = [12]) -> None:
        self._window = window
        self._favorable_months = favorable_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Extract the month of each session
        month_of_year = [int(date.fromisoformat(session_date).month) for session_date in closes["session_date"].to_list()]

        # Count how many favorable months are present for each stock
        favorable_count: dict[str, int] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            favorable_count[symbol] = sum(1 for m in month_of_year if m in self._favorable_months)

        # Filter out symbols with no favorable months
        picks = [symbol for symbol, count in favorable_count.items() if count > 0]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest