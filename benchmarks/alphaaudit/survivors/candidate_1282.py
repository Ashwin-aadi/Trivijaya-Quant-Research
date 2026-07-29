from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Historical data might exhibit seasonal patterns where certain stocks perform better "
        "at specific times of the year. This strategy aims to identify such trends and allocate "
        "capital accordingly."
    )

    def __init__(self, window: int = 365, seasonality_window: int = 90) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        seasonality_df = view.closes(lookback=self._seasonality_window)

        if seasonality_df.width == 0:
            return Signal(information_available_at=stamp, weights={})

        symbol_closes = {symbol: seasonality_df[symbol].to_list() for symbol in view.symbols}
        seasonal_trends: dict[str, float] = {}

        for symbol, closes_list in symbol_closes.items():
            if len(closes_list) < self._seasonality_window:
                continue
            trend = max([c / l - 1.0 for c, l in zip(closes[-self._seasonality_window:], closes[:-self._seasonality_window])])
            seasonal_trends[symbol] = trend

        top_symbols = sorted(seasonal_trends.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest