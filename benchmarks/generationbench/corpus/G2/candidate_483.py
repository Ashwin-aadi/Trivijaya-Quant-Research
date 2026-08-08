from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are typically more efficient in price discovery and have lower trading costs. "
        "By equally weighting liquid stocks, the strategy aims to benefit from reduced market impact during trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily liquidity for each symbol
        liquidity = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity")
            )
            .sort("liquidity", descending=True)
            .select(["symbol", "liquidity"])
        )

        # Filter top liquid symbols
        liquid_symbols = liquidity.head(10)["symbol"].to_list()

        if not liquid_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquid_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquid_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest