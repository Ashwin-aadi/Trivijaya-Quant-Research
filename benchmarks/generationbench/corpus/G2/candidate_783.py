from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are typically more efficient and less prone to price manipulation. "
        "By equal-weighting the most liquid stocks in the market, we aim to benefit from "
        "reduced transaction costs and potentially higher returns."
    )

    def __init__(self, min_volume: int = 1000000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=None)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_series = history.filter(pl.col("symbol").is_in(view.symbols)) \
                               .group_by("symbol") \
                               .agg((pl.col("volume").sum()).alias("total_volume")) \
                               .sort("total_volume", descending=True)

        filtered_symbols = [row["symbol"] for row in volume_series.iter_rows() if row["total_volume"] >= self._min_volume]
        weights = {s: 1.0 / len(filtered_symbols) for s in filtered_symbols}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest