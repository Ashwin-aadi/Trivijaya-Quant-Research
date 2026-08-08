from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in India exhibit seasonal trends due to calendar effects such as "
        "quarterly results announcements or government policies. By identifying and "
        "trading on these trends, we can capture excess returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the average close price and the maximum close price within the window
            avg_close = sum(values[-self._window // 2 :]) / (self._window // 2)
            max_close = max(values[-self._window // 2 :])

            # Check if this symbol shows a strong seasonal trend
            if values[-1] >= 0.95 * max_close and avg_close < max_close:
                seasonal_trends[symbol] = (max_close - avg_close) / max_close

        if not seasonal_trends:
            return Signal(information_available_at=stamp, weights={})

        # Sort symbols by the strength of their trend
        sorted_symbols = sorted(seasonal_trends.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_symbols[:3]]

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