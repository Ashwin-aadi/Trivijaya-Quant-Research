from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High-liquidity stocks are more efficient in price discovery and less prone to abnormal "
        "price movements. By equally weighting high-liquidity stocks, we aim to benefit from their "
        "greater market efficiency."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter symbols to only include those with sufficient liquidity
        min_volume_threshold = 100_000
        high_volume_symbols = history["symbol"].filter(
            (pl.col("volume") > min_volume_threshold).all()
        ).to_list()

        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation to selected symbols
        equal_weight = 1.0 / len(high_volume_symbols)
        signal_weights = {symbol: equal_weight for symbol in high_volume_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest