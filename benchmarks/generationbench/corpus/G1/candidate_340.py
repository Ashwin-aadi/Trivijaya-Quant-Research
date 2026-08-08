from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying symbols that have significantly "
        "deviated from their trailing average and then reverting back towards it, we can identify "
        "overbought or oversold conditions that may lead to a reversal in price."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing average and standard deviation
        avg_df = history.group_by("symbol").agg(
            (pl.col("adj_close").mean().alias("avg"), pl.col("adj_close").std().alias("std_dev"))
        )
        merged_history = history.join(avg_df, on="symbol", how="inner")
        merged_history = merged_history.with_columns(
            (
                (pl.col("adj_close") - pl.col("avg")) / pl.col("std_dev").abs()
            ).alias("z_score")
        )

        # Filter for symbols with high z-scores
        top_symbols = (
            merged_history.sort("z_score", descending=True)
            .select(pl.col("symbol"))
            .head(5)["symbol"]
            .to_list()
        )
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weights to selected symbols
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest