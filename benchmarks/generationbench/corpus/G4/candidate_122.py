from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Large market volatilities often precede significant trend reversals or continuation. "
        "This strategy scales positions based on realized volatility to capture trends while minimizing noise trading."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = []
        for symbol in view.symbols:
            high_low_range = (
                history.select(pl.col("high") - pl.col("low"))
                .with_columns((pl.col("high") - pl.col("low")).mean().alias("volatility"))
                .select("volatility")
                .to_series()
                .to_list()
            )
            if len(high_low_range) < self._window:
                continue
            volatilities.append({"symbol": symbol, "volatility": max(high_low_range)})

        volatility_df = pl.DataFrame(volatilities)
        volatility_ranked = (
            volatility_df.sort("volatility", descending=True).head(self._top_n)
        )

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in volatility_ranked["symbol"].to_list():
                continue
            weight = 1.0 / len(volatility_ranked)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest