from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures we allocate to stocks that are more active and "
        "liquid, reducing the risk of market impact from trading. Equal weighting across "
        "these stocks promotes diversification and reduces concentration risk."
    )

    def __init__(self, liquidity_threshold: int = 10_000_000) -> None:
        self._threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by volume
        filtered_history = history.filter(
            (pl.col("volume") > self._threshold) &
            pl.col("symbol").is_in(view.symbols)
        )

        if filtered_history.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation
        symbols = [str(symbol) for symbol in view.symbols if str(symbol) in history.columns]
        n_symbols = len(symbols)
        weight = 1.0 / n_symbols

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest