from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects the most liquid stocks in the Indian market to ensure broad "
        "diversification and periodic rebalancing to capitalize on market inefficiencies. "
        "High liquidity ensures that trades can be executed without significantly affecting stock prices."
    )

    def __init__(self, window: int = 30, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volume_history = history.select(
            pl.col("symbol"), (pl.col("volume") / pl.col("adj_close").sum().over("symbol")) * 100.0
        ).filter(pl.col("volume") > 0.1)

        liquidity_scores = (
            volume_history.groupby("symbol")
            .agg(
                pl.sum("volume").alias("total_volume"),
                (pl.col("volume").rolling_mean(window_size=self._window)).alias("avg_vol"),
                ((pl.col("adj_close").rolling_max(window_size=self._window) - pl.col("adj_close")) / pl.col("adj_close") * 100.0).alias("volatility"),
            )
            .sort("avg_vol", descending=True)
            .filter(pl.col("total_volume").sum() > 0.95 * history.height)
        )

        if liquidity_scores.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = liquidity_scores.head(self._top_n)["symbol"].to_list()
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