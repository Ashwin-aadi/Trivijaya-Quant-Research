from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are less prone to price manipulation and may offer more stable returns. "
        "By equal weighting these stocks, we can benefit from the market's increased buying and selling pressure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(
                (pl.col("volume").rolling_sum(window=self._window) > 0)
                & (pl.col("adj_close").rolling_mean(window=self._window) > 0)
            )
            .with_columns(
                (pl.col("volume") / pl.col("volume").sum().over("symbol")).alias("scaled_volume")
            )
            .group_by("symbol", maintain_order=True)
            .agg(pl.col("scaled_volume").mean().alias("average_scaled_volume"))
        )

        top_symbols = (
            liquidity_screened.sort("average_scaled_volume", descending=True).select(
                pl.col("symbol")
            ).head(10)["symbol"].to_list()
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