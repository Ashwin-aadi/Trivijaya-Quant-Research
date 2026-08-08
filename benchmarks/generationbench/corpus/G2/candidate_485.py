from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "A breakout followed by a continuation trend can indicate strong momentum and potential "
        "for further price increases. This strategy aims to capitalize on such trends."
    )

    def __init__(self, window: int = 20, min_trend_length: int = 5) -> None:
        self._window = window
        self._min_trend_length = min_trend_length

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._min_trend_length - 1)

        if history.height < self._window + self._min_trend_length - 1:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, list[float]] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            trend = _check_trend(adj_closes)
            if trend is not None:
                trends[symbol] = trend

        valid_symbols = [
            s for s, t in trends.items() if len(t) >= self._min_trend_length
        ]
        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(valid_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in valid_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _check_trend(values: list[float]) -> list[float] | None:
    trend_direction = 0
    trend = []
    for i in range(1, len(values)):
        if values[i] > values[i - 1]:
            if trend_direction == -1 or (trend and trend[-1] < 0):
                trend.clear()
            trend_direction = 1
        elif values[i] < values[i - 1]:
            if trend_direction == 1 or (trend and trend[-1] > 0):
                trend.clear()
            trend_direction = -1
        else:
            continue
        trend.append(trend_direction)
    return trend if len(trend) >= _min_trend_length else None