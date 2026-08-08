from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain periods of the year may exhibit stronger trends or higher returns in specific sectors. "
        "This strategy aims to identify and capitalize on these seasonal effects by overweighting symbols that have historically performed well during certain months."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 365)
        if history.height < self._window * 365:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(closes) < self._window * 365:
                continue
            monthly_returns: list[float] = []
            for i in range(len(closes) - 1):
                month = (history["session_date"][i].month - 1) % 12
                next_month = (history["session_date"][i + 1].month - 1) % 12
                if month == next_month:
                    monthly_returns.append((closes[i + 1] - closes[i]) / closes[i])
            avg_return = sum(monthly_returns) / len(monthly_returns)
            seasonal_trends[symbol] = avg_return

        top_symbols = sorted(seasonal_trends, key=seasonal_trends.get, reverse=True)[: self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest