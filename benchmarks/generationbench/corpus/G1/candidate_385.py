from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Mean reversion theory suggests that prices and values of assets will eventually "
        "return to the long-term mean. By identifying stocks that have deviated significantly "
        "from their historical mean, one can exploit this phenomenon for profitable trades."
    )

    def __init__(self, window: int = 5, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            closes.lazy()
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("m"))
            .collect()
        )
        std_devs = (
            closes.lazy()
            .group_by("symbol")
            .agg(pl.col("adj_close").stddev().alias("s"))
            .collect()
        )

        mean_reversion_signals = (
            closes
            .join(mean_prices, on="symbol", how="inner")
            .join(std_devs, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") - pl.col("m")) / (2 * pl.col("s")).alias("z_score")
            )
        )

        thresholded_signals = mean_reversion_signals.with_columns(
            (pl.when(pl.col("z_score").abs() > self._threshold).then(1)
             .otherwise(0)).alias("revert_signal")
        )

        symbols_with_signals = thresholded_signals.select("symbol", "revert_signal").filter(
            pl.col("revert_signal") == 1
        ).select("symbol").to_series().to_list()

        if not symbols_with_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest