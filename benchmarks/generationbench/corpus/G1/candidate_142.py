from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data suggests that certain stocks in the NIFTY 100 exhibit stronger "
        "performance during specific times of the year. By identifying these seasonal trends,"
        " we can construct a strategy to capitalize on such patterns."
    )

    def __init__(self, window: int = 365, threshold: float = 0.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = set(history["symbol"].to_list())
        filtered_history = (
            history.filter(pl.col("session_date").dt.month().is_in([12, 1, 2]))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(365) - 1.0).alias("return_ratio"),
            )
        )

        filtered_history = (
            filtered_history.with_columns(
                pl.when(pl.col("return_ratio") > self._threshold)
                .then(1)
                .otherwise(0)
                .alias("seasonal_signal")
            )
        ).filter(pl.col("seasonal_signal").is_not_null())

        if filtered_history.height < 3:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            filtered_history.group_by("symbol")
            .agg(
                (pl.col("return_ratio") * pl.col("seasonal_signal")).sum().alias("weighted_return"),
            )
            .sort(pl.col("weighted_return"), descending=True)
            .limit(5)["symbol"]
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest