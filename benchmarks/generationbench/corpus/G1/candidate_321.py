from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term. "
        "By tilting our portfolio towards low volatility, we aim to capture this excess return."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_series = (
            history.select(pl.col("symbol"))
            .join(history.lazy().group_by("symbol").agg(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean()
            ).collect(), on="symbol", how="left")
            .select(pl.col("symbol"), "stddev")
        )
        if volatility_series.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [row[0] for row in volatility_series.sort("stddev").rows()]
        weight = 1.0 / min(20, len(symbols))
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