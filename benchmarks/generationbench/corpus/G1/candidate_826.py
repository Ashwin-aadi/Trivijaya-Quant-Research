from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance at specific times of the year due to seasonal "
        "effects. By identifying these patterns, we can allocate capital more effectively during "
        "favorable periods."
    )

    def __init__(self, window: int = 365, seasonality_periods: list[tuple[str, int]] = []) -> None:
        self._window = window
        self._seasonality_periods = seasonality_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("symbol").alias("close"))

        seasonality_scores = {s: 0.0 for s in self._seasonality_periods}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            date_field = history.select(pl.col("session_date")).to_numpy().flatten()
            dates = [date.fromisoformat(d.decode()) for d in date_field]

            seasonal_scores = {s: 0.0 for s in self._seasonality_periods}
            for i, (period_name, period_length) in enumerate(self._seasonality_periods):
                start_date = stamp - date(1, 1, 1) + date.fromisoformat(period_name)
                end_date = start_date + date(1, 1, 1) * period_length
                season_indices = [j for j, d in enumerate(dates) if start_date <= d < end_date]
                seasonal_scores[period_name] = sum(values[j] for j in season_indices)

            max_score = max(seasonal_scores.values())
            best_period = next(k for k, v in seasonal_scores.items() if v == max_score)
            seasonality_scores[best_period] += 1.0

        top_period = max(seasonality_scores, key=seasonality_scores.get)
        picks = [symbol for symbol in view.symbols if symbol in closes.columns]
        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: seasonality_scores[s] * weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest