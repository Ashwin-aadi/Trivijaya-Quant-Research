from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screens can help in reducing transaction costs and identifying "
        "more stable assets. Equal weighting ensures that no single stock dominates the portfolio."
    )

    def __init__(self, liquidity_threshold: float = 1_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by liquidity
        filtered_history = history.filter(
            (pl.col("volume") > self._threshold) &
            (pl.col("symbol").is_in(view.symbols))
        )
        
        symbols = filtered_history["symbol"].unique().to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest