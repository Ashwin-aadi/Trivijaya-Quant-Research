from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Prices that deviate significantly from their trailing mean are likely to revert "
        "to the mean. This is a common empirical finding in financial markets."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.select(pl.col("adj_close").mean().alias("trailing_mean"))
            .collect()
            .get_column("trailing_mean")
            .to_list()[0]
        )
        symbols_to_trade = [
            symbol
            for symbol in view.symbols
            if abs(view.latest_close()[symbol] - mean_close) > 1.5 * (
                closes.select(pl.col(symbol).std()).collect().get_column(symbol).to_list()[0]
            )
        ]

        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols_to_trade},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest