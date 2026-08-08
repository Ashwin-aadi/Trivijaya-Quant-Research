from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines the relative strength of a stock with its recent volatility. "
        "Stocks that are relatively strong and have low recent volatility may offer better entry points."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength (RS) as the ratio of close to 20-day moving average
        sma = history["adj_close"].mean().over(pl.arange(1, self._window + 1)).alias("sma")
        rs = history["adj_close"] / sma

        # Calculate volatility using standard deviation over a smaller window
        vol = (history.select(
            (pl.col("adj_close") - pl.col("adj_close").mean().over(pl.arange(1, self._volatility_window + 1))).pow(2)
        ).select((pl.all().sum()).sqrt()))

        # Filter out symbols with insufficient data
        filtered_symbols = [s for s in view.symbols if s in rs.columns and not rs[s].is_null().any()]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Rank the symbols based on RS adjusted for volatility
        ranked = (rs.select(
            pl.col("symbol").alias("symbol"),
            (rs["adj_close"] / vol).rank(method="dense", descending=True).cast(pl.Float64)
        ).sort("symbol", descending=True)).select("symbol")

        top_symbols = [str(ranked[i, 0]) for i in range(min(self._window, len(filtered_symbols)))]
        weight = 1.0 / len(top_symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest