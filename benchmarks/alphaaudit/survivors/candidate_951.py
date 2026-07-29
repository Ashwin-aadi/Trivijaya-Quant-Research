from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that we only invest in stocks with sufficient trading "
        "volume. Equal weighting among the selected symbols provides a simple and fair way to "
        "distribute capital across the chosen assets."
    )

    def __init__(self, min_volume: int = 100_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by minimum volume
        filtered_history = (
            history.filter(pl.col("volume") > self._min_volume)
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
        )

        if filtered_history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(symbol) for symbol in filtered_history.head(5)["symbol"]]
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