from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy screens for stocks with high liquidity and then allocates capital "
        "equally among them to capture the market-wide trend without overconcentration in low-liquidity securities."
    )

    def __init__(self, min_volume: int = 1_000_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Consider a full year of data
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_symbols = [
            symbol for symbol in view.symbols
            if float(history[symbol]["volume"].sum()) >= self._min_volume
        ]

        if len(liquidity_screened_symbols) < 5:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in liquidity_screened_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest