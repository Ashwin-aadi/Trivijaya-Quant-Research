from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity plays a crucial role in market efficiency. Equally weighting stocks based on "
        "their liquidity can help identify the most liquid and active stocks for investment."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 1:
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = {symbol: float(volume.sum()) for symbol, volume in
                            (closes.select(pl.col("symbol"), pl.col("volume").sum()).to_dicts())}
        total_volume = sum(liquidity_scores.values())
        if not total_volume:
            return Signal(information_available_at=stamp, weights={})

        weights = {s: l / total_volume for s, l in liquidity_scores.items()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest