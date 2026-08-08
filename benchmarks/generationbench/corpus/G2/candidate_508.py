from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion strategies exploit mean-reverting behavior in financial time series. "
        "When a stock's price has deviated significantly from its historical average, "
        "it often tends to revert back towards that mean over time. This can be harnessed for "
        "trading profits by going long on stocks that have fallen below their trailing average "
        "and shorting those that have risen above it."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not view.symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean_price"))
        )

        latest_closes = pl.DataFrame(
            {
                "symbol": [s for s in view.symbols if s in closes.columns],
                "latest_close": [float(view.latest_close()[s]) for s in view.symbols]
            }
        )

        merged_data = mean_prices.join(latest_closes, on="symbol")
        reversion_scores = (
            (merged_data["latest_close"] - merged_data["mean_price"])
            .to_list()
        )
        symbols = [s for s in merged_data["symbol"].to_list()]
        
        long_symbols = [
            symbol
            for score, symbol in zip(reversion_scores, symbols)
            if score < 0
        ]
        short_symbols = [
            symbol
            for score, symbol in zip(reversion_scores, symbols)
            if score > 0
        ]

        long_weight = 1.0 / len(long_symbols) if long_symbols else 0
        short_weight = -1.0 / len(short_symbols) if short_symbols else 0

        return Signal(
            information_available_at=stamp,
            weights={
                **({symbol: long_weight for symbol in long_symbols}),
                **({symbol: short_weight for symbol in short_symbols})
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest