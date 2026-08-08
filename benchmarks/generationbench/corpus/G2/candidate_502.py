from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Seasonality(Strategy):
    rationale = (
        "Certain stocks in India may exhibit seasonality effects, where their performance "
        "is consistently higher during specific times of the year. This could be due to"
        " seasonal demand patterns or other calendar-related events."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
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

            # Calculate returns
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            # Identify months for each session
            month_dates = [date.fromordinal(int(d)) for d in view.closes().column("session_date").to_list()]
            monthly_returns = {month.date(): sum(r for _, r in zip(month_dates, returns) if date.fromordinal(int(_)).month == month.month) for month in set([d.toordinal() for d in month_dates])}

            # Find the months with highest average return
            top_months = sorted(monthly_returns.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
            if any(date.fromordinal(int(d)).month == m.month for d, _ in monthly_returns.items() for m, _ in top_months):
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