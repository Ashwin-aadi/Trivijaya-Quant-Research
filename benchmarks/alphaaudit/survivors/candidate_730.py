from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Historical data from the Indian market suggests that certain stocks exhibit "
        "seasonal patterns in their performance. By identifying periods when these "
        "stocks historically outperform, we can construct a strategy to capitalize on "
        "these seasonal trends."
    )

    def __init__(self, window: int = 100, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            avg_close = sum(values) / len(values)
            recent_avg_close = sum(values[-21:]) / 21  # 3 weeks of data
            seasonal_factor = (recent_avg_close - avg_close) / avg_close
            seasonal_factors[symbol] = seasonal_factor

        sorted_factors = sorted(seasonal_factors.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_factors[: self._top_n]]
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