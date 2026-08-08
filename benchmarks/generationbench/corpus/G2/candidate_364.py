from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion strategies assume that prices will return to historical levels of "
        "mean or median after a deviation. In the context of the NIFTY 100, this means "
        "buying stocks that have fallen below their recent mean price level and selling those "
        "that have risen above it."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]

        mean_close = (
            history.select(pl.col("adj_close").mean().over("symbol"))
            .with_columns(
                pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0
            )
            .sort("session_date", descending=True)
            .select([pl.col("adj_close"), "session_date"])
            .filter(pl.col("session_date") == view.as_of)
            .collect()
        )["adj_close"].to_list()[0]

        deviations = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / mean_close - 1.0).alias("deviation")
            )
            .sort("symbol", descending=True)
            .select(["symbol", "deviation"])
        )

        top_bullish = [row[0] for row in deviations.scan().filter(pl.col("deviation") > 0.25).collect().rows()]
        bottom_bearish = [row[0] for row in deviations.scan().filter(pl.col("deviation") < -0.25).collect().rows()]

        weights: dict[str, float] = {}
        if top_bullish:
            weight = 1.0 / len(top_bullish)
            for symbol in top_bullish:
                weights[symbol] = weight
        elif bottom_bearish:
            weight = -1.0 / len(bottom_bearish)
            for symbol in bottom_bearish:
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest