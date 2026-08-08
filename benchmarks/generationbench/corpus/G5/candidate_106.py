from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the Indian stock market can be attributed to various factors such "
        "as festival effects or corporate actions. By identifying stocks that exhibit strong "
        "seasonal trends, we aim to capture these predictable price movements for profit."
    )

    def __init__(self, window: int = 60, seasonal_periods: tuple[int, ...] = (30,)) -> None:
        self._window = window
        self._seasonal_periods = seasonal_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_seasonality_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            scores = [0.0] * len(closes)
            valid_periods = []
            for period in self._seasonal_periods:
                if period > len(closes):
                    break
                seasonal_trend = (closes[-period:] - closes[:-period]) / closes[:-period]
                if not any(st < 0 for st in seasonal_trend):  # Avoid division by zero or small numbers
                    valid_periods.append(period)
            scores[valid_periods[-1] - 1 : -valid_periods[0] + 1] = [
                max(st, 0) for st in seasonal_trend if not any([st < 0, st == 0])  # Filter out negative or zero values
            ]
            avg_score = sum(scores) / max(len(scores), 1)
            symbol_seasonality_scores[symbol] = avg_score

        top_symbols = sorted(symbol_seasonality_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest