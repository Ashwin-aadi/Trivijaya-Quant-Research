from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks in the Indian market may exhibit stronger performance during specific "
        "calendar periods. This strategy identifies such periods and allocates capital to "
        "performing symbols based on historical seasonality."
    )

    def __init__(self, window: int = 365, lookback: int = 2) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback - 1)
        if history.height < self._window + self._lookback - 1:
            return Signal(information_available_at=stamp, weights={})

        seasonal_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            mean_close = sum(closes[-self._window:]) / self._window
            last_period_close = closes[-1]
            score = (last_period_close - mean_close) / mean_close
            seasonal_scores[symbol] = score

        sorted_symbols = [
            s for _, s in sorted(seasonal_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        picks = sorted_symbols[: self._lookback]
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