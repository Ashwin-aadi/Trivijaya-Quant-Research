from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and historical returns will eventually move "
        "towards the long-term mean. In a short horizon, extreme deviations from this mean can "
        "indicate potential reversals. Identifying such deviations in the Indian market (NIFTY 100) "
        "can generate trading opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(pl.col("adj_close").mean().alias("mean"))
        z_scores = (
            closes.join(mean_close, on="session_date")
            .with_column(
                (pl.col("adj_close") - pl.col("mean")).abs()
                / (pl.col("adj_close") - pl.col("mean")).std().over("symbol").alias("z_score")
            )
        )

        extreme_deviations = z_scores.filter(pl.col("z_score") > self._threshold)
        if extreme_deviations.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in extreme_deviations.rows()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest