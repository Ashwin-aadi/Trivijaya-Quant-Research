from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "By exploiting historical seasonality in Indian equity markets, this strategy "
        "identifies stocks with positive returns during specific months. This is based on the "
        "historical observation that certain months exhibit stronger performance due to factors like "
        "earnings announcements and regulatory calendars."
    )

    def __init__(self, window: int = 60, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(history.columns) != 21:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        monthly_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            daily_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            month_dates = [
                d for d in daily_closes[0::21]
                if isinstance(d, date) and 2019 <= d.year <= 2023
            ]
            monthly_closes = [daily_closes[i * 21 + 20] for i in range(len(month_dates))]

            avg_monthly_return = sum(
                (close - prev_close) / prev_close
                for close, prev_close in zip(monthly_closes[1:], monthly_closes[:-1])
            )
            monthly_returns[symbol] = avg_monthly_return

        top_symbols = sorted(monthly_returns.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest