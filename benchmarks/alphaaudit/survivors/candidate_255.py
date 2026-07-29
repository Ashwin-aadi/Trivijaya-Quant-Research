from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key indicator of market interest. By selecting the most liquid "
        "stocks and applying equal weighting to them, we can ensure that our portfolio "
        "includes stocks with high trading volumes, reducing the risk of significant bid-ask spreads."
    )

    def __init__(self, window: int = 20, liquidity_threshold: float = 1e6) -> None:
        self._window = window
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        filtered_history = (
            history.filter(
                (pl.col("symbol").is_in(view.symbols))
                & (pl.col("volume").gt(self._liquidity_threshold))
            )
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
            )
        )

        if filtered_history.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = (
            filtered_history.sort(
                "total_volume", descending=True
            ).select(pl.col("symbol")).to_series().to_list()[:10]
        )

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest