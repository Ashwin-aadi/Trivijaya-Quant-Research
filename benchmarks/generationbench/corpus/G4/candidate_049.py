from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their historical "
        "price levels due to transient factors and are expected to revert towards their long-term means. "
        "The short-horizon mean reversion in the Indian equity market is exploited by selecting stocks with sharp deviations, "
        "entering trades when prices approach or cross their respective moving averages."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_10 = (
            history.groupby("symbol")
                   .agg((pl.col("adj_close").shift(-self._window).mean().over(pl.col("session_date"))).alias("sma"))
        )
        history = history.join(sma_10, on="symbol", how="inner")

        deviations = (history["adj_close"] - history["sma"]).abs()
        ranked_symbols = deviations.sort(by="sma", descending=True).select(["symbol", "session_date", "adj_close", "sma", "deviations"])[:self._top_n]

        weights = {row.symbol: 1.0 / len(ranked_symbols) for row in ranked_symbols.rows()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest