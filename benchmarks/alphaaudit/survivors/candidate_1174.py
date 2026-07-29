from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance at specific times of the year. "
        "By identifying and leveraging these seasonal patterns, we can generate "
        "profitable trading signals."
    )

    def __init__(self, season_length: int = 120, top_n: int = 5) -> None:
        self._season_length = season_length
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._season_length)
        if closes.height < self._season_length:
            return Signal(information_available_at=stamp, weights={})

        seasonality_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._season_length:
                continue

            # Calculate the average return over the season length
            mean_return = sum((v / values[i - 1] - 1.0 for i, v in enumerate(values[1:])), start=0.0)
            seasonality_scores[symbol] = mean_return

        # Sort symbols by their seasonality scores and select top N performers
        sorted_symbols = sorted(seasonality_scores.items(), key=lambda x: x[1], reverse=True)
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