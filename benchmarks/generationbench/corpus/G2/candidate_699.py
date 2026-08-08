from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more efficiently priced and have lower trading costs. "
        "By equal-weighting these stocks, we aim to benefit from the reduced trading frictions "
        "and potentially higher market-wide average returns."
    )

    def __init__(self, min_volume: int = 100_000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=50)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_liquidity_symbols = []
        for symbol in view.symbols:
            daily_volumes = history.select(pl.col(symbol)).to_numpy().ravel()
            daily_volumes = [float(v) for v in daily_volumes if not pl.col("close").is_nan()]
            if len(daily_volumes) < 20 or any(volume <= self._min_volume for volume in daily_volumes):
                continue
            high_liquidity_symbols.append(symbol)

        weights = {symbol: 1.0 / len(high_liquidity_symbols) for symbol in high_liquidity_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest