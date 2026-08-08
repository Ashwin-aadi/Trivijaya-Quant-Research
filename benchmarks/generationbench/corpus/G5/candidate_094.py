from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Seasonal effects can significantly impact stock prices. This strategy "
        "exploits historical patterns by identifying stocks that have historically "
        "performed well during certain times of the year."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors = {symbol: {} for symbol in view.symbols}

        for session_date in set(closes["session_date"].to_list()):
            dates_in_window = [date.fromordinal(d.to_timestamp("D").date().toordinal()) for d in closes[closes["session_date"] == session_date].columns[1:]]
            if len(dates_in_window) < self._window:
                continue

            month_of_year = [d.month for d in dates_in_window]
            mean_closes = [float(c) for c in closes[closes["session_date"] == session_date]["adj_close"].drop_nulls().to_list()]

            for i, symbol in enumerate(view.symbols):
                seasonality_factors[symbol][date.fromordinal(session_date.to_timestamp("D").date().toordinal())] = mean_closes[i]

        picks: list[str] = []
        for symbol in view.symbols:
            monthly_values = [seasonality_factors[symbol].get(d, 0.0) for d in sorted(seasonality_factors[symbol])]
            if len(monthly_values) < 12:
                continue

            max_value = max(monthly_values[-6:])
            picks.extend([s for s, v in seasonality_factors[symbol].items() if v == max_value])

        unique_picks = list(set(picks))
        weight = 1.0 / len(unique_picks) if unique_picks else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in unique_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest