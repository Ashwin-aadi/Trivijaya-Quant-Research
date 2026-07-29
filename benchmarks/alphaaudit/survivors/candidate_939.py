from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening helps identify stocks with higher trading volumes, which are "
        "expected to have lower bid-ask spreads and higher marketability. Equal weighting of "
        "these stocks can provide a diversified portfolio."
    )

    def __init__(self, liquidity_threshold: float = 10_000) -> None:
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.height < 365:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if "volume" in history.columns]
        filtered_history = history.filter(
            pl.col("symbol").is_in(symbols) & (pl.col("volume") > self._liquidity_threshold)
        )
        if filtered_history.height < len(symbols):
            return Signal(information_available_at=stamp, weights={})

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