from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are less prone to price manipulation and may offer more reliable "
        "returns. Equal weighting ensures no single stock dominates the portfolio, reducing risk."
    )

    def __init__(self, min_volume_threshold: int = 10_000) -> None:
        self._min_volume_threshold = min_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by volume
        filtered_history = history.filter(
            pl.col("volume") > self._min_volume_threshold
        )

        if filtered_history.height < 20:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting among the high liquidity stocks
        symbols = filtered_history["symbol"].to_list()
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest