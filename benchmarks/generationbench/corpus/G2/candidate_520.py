from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels revert to their mean over time. By identifying symbols that have "
        "deviated significantly from their trailing mean, we can exploit this reversion "
        "trend for profit."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing mean and standard deviation
        mean_df = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean().alias("mean"),
                         pl.col("adj_close").std().alias("std")))
        )
        latest_closes = view.closes(lookback=self._window)
        latest_prices = latest_closes.select(pl.all().to_list())

        # Compute z-scores for the latest close relative to the trailing mean
        combined_df = (
            pl.concat([mean_df, latest_prices]).lazy()
                   .with_columns(
                       (pl.col("adj_close") - pl.col("mean")) / pl.col("std").alias("zscore")
                   )
                   .collect()
        )

        # Identify symbols with z-scores above a certain threshold
        high_z_score_threshold = 1.5
        picks: list[str] = []
        for row in combined_df.iter_rows():
            symbol, _, mean, std, zscore = [row[i] for i in range(5)]
            if float(zscore) > high_z_score_threshold:
                picks.append(str(symbol))

        # Form a signal based on the identified symbols
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest