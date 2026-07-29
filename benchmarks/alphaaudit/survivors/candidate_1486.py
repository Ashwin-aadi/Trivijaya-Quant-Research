from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance during specific months or seasons. "
        "By identifying these patterns, we can allocate capital to perform better in those periods."
    )

    def __init__(self, window: int = 5, seasonality_window: int = 3) -> None:
        self._window = window
        self._seasonality_window = seasonality_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._seasonality_window)

        if closes.height < self._seasonality_window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_effects: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(values[-self._window:]) / self._window
            seasonality_effects[symbol] = (values[-1] - mean_close) / mean_close

        top_performers: list[str] = sorted(
            seasonality_effects, key=lambda k: seasonality_effects[k], reverse=True
        )[:2]

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest