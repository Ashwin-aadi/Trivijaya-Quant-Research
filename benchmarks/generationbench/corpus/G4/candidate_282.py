from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy exploits the phenomenon where stocks with higher relative strength "
        "outperform the broader market over time. By selecting a portfolio of top N (e.g., 10) "
        "stocks based on their RSI against the broad universe index, we aim to capitalize on "
        "sustained outperformance driven by positive market sentiment."
    )

    def __init__(self, window: int = 14, n_top_stocks: int = 10) -> None:
        self._window = window
        self._n_top_stocks = n_top_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate RSI for each stock
        rsi_frame = (
            history
            .select(
                pl.col("symbol"),
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                (pl.col("adj_close") - pl.col("adj_close").shift(self._window)).alias("total_change")
            )
            .group_by("symbol", maintain_order=True)
            .agg(
                pl.count().alias("count"),
                (pl.sum(pl.col("return")) / pl.col("count")).alias("mean_return"),
                ((pl.col("total_change") / self._window) - 1.0).alias("daily_change")
            )
            .with_columns(
                (
                    (pl.col("mean_return").abs() + pl.col("daily_change").abs())
                    / (2 * pl.col("mean_return").abs().rank(method="dense", descending=True))
                ).alias("rs"),
                100 - (100 / (1 + pl.col("rs"))).alias("rsi")
            )
        )

        if rsi_frame.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols by RSI and select top N
        ranked_symbols = (
            rsi_frame.sort("rsi", descending=True)
            .select(["symbol", "rsi"])
            .head(self._n_top_stocks)["symbol"]
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(ranked_symbols.to_list(), [weight] * len(ranked_symbols)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest