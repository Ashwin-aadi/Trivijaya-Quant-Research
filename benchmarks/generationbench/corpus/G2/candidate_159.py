from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and can indicate areas of the market where "
        "information flows more freely. High liquidity assets are typically seen as less risky "
        "and may offer better returns due to lower bid-ask spreads, making them attractive for "
        "equal-weighted portfolios."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volumes = history["volume"].to_list()
        symbols = [symbol for symbol in view.symbols if symbol in volumes]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volume_sum = sum(volumes)
        weights = {s: v / volume_sum for s, v in zip(symbols, volumes)}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest