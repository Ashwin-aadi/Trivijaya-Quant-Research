from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting top-performing stocks "
        "over the past 3-6 months. It aims to capture persistent price performance driven by "
        "investor sentiment and market expectations."
    )

    def __init__(self, lookback_period: int = 120, num_top_stocks: int = 20) -> None:
        self._lookback_period = lookback_period
        self._num_top_stocks = num_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.is_empty() or history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        # Calculate cumulative returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_period) - 1.0).alias("cumulative_return")
        )
        latest_close_df = view.closes(lookback=self._lookback_period)
        latest_closes = {s: float(v) for s, v in zip(latest_close_df["symbol"], latest_close_df["adj_close"])}
        
        # Rank stocks based on cumulative returns and fundamental filters
        history = (
            history.select(
                pl.col("symbol"),
                "cumulative_return",
                pl.col("adj_close").mean().alias("avg_price"),
                pl.col("volume").mean().alias("avg_volume")
            )
            .with_columns(
                (pl.col("cumulative_return") / 10 + pl.col("avg_volume") / 100 - pl.col("avg_price") / 100).alias("score")
            )
        )

        # Filter and select top stocks
        ranked_stocks = (
            history.sort("score", descending=True)
            .head(self._num_top_stocks)
            .select(pl.col("symbol"))
            .to_dict(as_series=False)["symbol"]
        )

        weights = {s: 1.0 / len(ranked_stocks) for s in ranked_stocks}
        
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest