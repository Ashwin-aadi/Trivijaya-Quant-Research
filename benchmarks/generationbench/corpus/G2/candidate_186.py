from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Companies that outperform their peers over a period of time may indicate "
        "stronger fundamentals or better management. This strategy aims to identify "
        "such companies by comparing the relative strength of each stock against all others."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if view.history().is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback)

        # Calculate the average close for each symbol over the lookback period
        avg_closes = (
            closes.groupby("symbol")
                  .agg(pl.col("adj_close").mean().alias("avg_close"))
        )

        # Calculate the relative strength as the ratio of current close to the mean close
        rel_strength = (
            view.closes()
                   .join(avg_closes, on="symbol", how="left")
                   .with_columns(
                       (pl.col("adj_close") / pl.col("avg_close").fill_null(1)).alias("rel_strength")
                   )
                   .select(["session_date", "symbol", "rel_strength"])
        )

        # Get the top N symbols by relative strength
        top_symbols = rel_strength.sort("rel_strength", descending=True).select(["symbol"])[:5]

        if top_symbols.height < 5:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / top_symbols.height
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols["symbol"].to_list()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest