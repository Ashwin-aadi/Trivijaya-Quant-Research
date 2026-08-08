from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market interest and confidence in an asset. "
        "Highly liquid assets are more likely to be overvalued or mispriced, offering opportunities for value investors."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_symbols = [
            symbol
            for symbol in view.symbols
            if (
                (symbol not in history.select("volume").is_null().columns)
                and history[symbol].height >= self._lookback
            )
        ]

        if not liquidity_screened_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_screened_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_screened_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest