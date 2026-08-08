from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the anomaly in equity markets where smaller, less liquid stocks "
        "often exhibit higher returns due to underpriced information. By selecting and equally "
        "weighting the most liquid stocks from the Indian stock market, we aim to capture enhanced "
        "returns while maintaining diversification and minimizing concentration risk."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = (
            history.select(
                pl.col("symbol"), (pl.col("volume").sum() / self._window).alias("avg_volume")
            )
            .sort("avg_volume", descending=True)
            .head(self._top_n)[["symbol"]]
        )

        if volume_df.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [str(row[0]) for row in volume_df.rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest