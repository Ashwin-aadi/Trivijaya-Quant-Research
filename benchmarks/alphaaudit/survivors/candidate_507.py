from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered for the portfolio. Equal weighting across these stocks can provide a balanced "
        "exposure and potentially mitigate concentration risk."
    )

    def __init__(self, liquidity_threshold: float = 1_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Consider the last year of data
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_filter = history.filter(
            (pl.col("symbol").is_in(view.symbols))
            & (pl.col("volume") >= self._threshold)
        )
        if volume_filter.height < 252:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol) for symbol in view.symbols]
        weight = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest