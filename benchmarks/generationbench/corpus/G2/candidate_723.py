from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are less prone to price manipulation and may have more stable "
        "prices. Equal weighting ensures that no single stock dominates the portfolio, thus "
        "reducing the risk of overconcentration in low-liquidity stocks."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols with insufficient trading volume
        min_volume = 100_000
        filtered_history = (
            history.filter(pl.col("volume") > min_volume)
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_close"),
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
        )

        # Get the top symbols with sufficient liquidity
        top_symbols = filtered_history.head(10)["symbol"].to_list()

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