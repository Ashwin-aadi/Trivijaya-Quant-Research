from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data often reveal that certain stocks exhibit stronger performance "
        "during specific months of the year. By identifying such patterns, we can leverage "
        "seasonality to make informed trading decisions."
    )

    def __init__(self, window: int = 60, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            seasonality_scores[symbol] = (
                pl.DataFrame(values)
                .with_column(
                    (pl.col("value") > pl.col("value").shift(30)).alias("above_30_day")
                )
                .group_by("symbol")
                .agg((pl.col("above_30_day").sum() / 2).alias("seasonality_score"))
                .select("seasonality_score")
                .to_series()
                .to_list()[0]
            )

        top_performing_symbols = [
            symbol for symbol, score in sorted(
                seasonality_scores.items(), key=lambda item: -item[1]
            )
            if score > self._threshold
        ][:5]

        if not top_performing_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performing_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_performing_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest