from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered. Equal weighting among these securities can provide a diversified and "
        "potentially stable investment strategy."
    )

    def __init__(self, min_volume: int = 1000000) -> None:
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=252)  # Consider a lookback of one year for liquidity
        if history.height < 252:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_history = history.filter(pl.col("volume") >= self._min_volume)
        if filtered_history.height == 0 or len(symbols) < 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the liquidity-weighted equal weights
        volumes = [float(v) for v in filtered_history["volume"].to_list()]
        total_volume = sum(volumes)
        if total_volume == 0:
            return Signal(information_available_at=stamp, weights={})
        
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