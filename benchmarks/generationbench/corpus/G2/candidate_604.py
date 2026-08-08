from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion strategies are based on the empirical observation that assets often "
        "reverse direction after extreme price movements. By identifying symbols with large"
        " deviations from their recent mean prices, we can exploit such reversions for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Compute the mean price over the lookback period for each symbol
        mean_prices = (
            closes.select(pl.col("symbol").alias("symbol"), pl.col("adj_close"))
                   .group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_price"))
                   .collect()
        )

        # Compute the deviations from the mean price for each close
        deviations = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            mean_price = float(mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item())
            current_close = float(closes[symbol][-1])
            deviation = (current_close - mean_price) / mean_price
            deviations.append((symbol, deviation))

        # Identify symbols with the largest positive and negative deviations
        top_reversals: list[str] = []
        for symbol, deviation in sorted(deviations, key=lambda x: abs(x[1]), reverse=True):
            if len(top_reversals) >= 5:
                break
            top_reversals.append(symbol)

        if not top_reversals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_reversals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_reversals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest