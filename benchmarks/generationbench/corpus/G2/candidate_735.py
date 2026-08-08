from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are generally more efficient in price discovery and less prone to "
        "abnormal price movements. By equal-weighting these highly liquid stocks, the strategy aims "
        "to benefit from the stability and efficiency of the market's pricing mechanism."
    )

    def __init__(self, min_trading_volume: float = 100_000) -> None:
        self._min_trading_volume = min_trading_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_liquid_symbols = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg(
                pl.sum("volume").alias("total_volume"),
                pl.count().alias("trading_days"),
            )
            .filter((pl.col("total_volume") > self._min_trading_volume) & (pl.col("trading_days") >= 180))
            .select(pl.col("symbol"))
        )

        if high_liquid_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol[0]) for symbol in high_liquid_symbols.to_numpy()]
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