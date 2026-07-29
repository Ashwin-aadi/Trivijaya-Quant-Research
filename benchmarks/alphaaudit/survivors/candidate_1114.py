from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks may exhibit stronger performance during specific seasons or months "
        "of the year due to economic activities or investor behavior. This strategy aims to "
        "identify such trends and allocate capital accordingly."
    )

    def __init__(self, seasonal_window: int = 60, top_n: int = 5) -> None:
        self._seasonal_window = seasonal_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._seasonal_window)
        if closes.height < self._seasonal_window:
            return Signal(information_available_at=stamp, weights={})

        symbol_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._seasonal_window:
                continue

            mean_close = sum(values) / len(values)
            seasonal_high_count = sum(1 for v in values if v >= 0.95 * mean_close)
            seasonal_low_count = sum(1 for v in values if v <= 0.95 * mean_close)

            score = (seasonal_high_count - seasonal_low_count) / self._seasonal_window
            symbol_scores[symbol] = score

        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]

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