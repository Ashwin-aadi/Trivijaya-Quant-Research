from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion to the mean suggests that after a significant move, prices will "
        "eventually return to their historical average. This strategy identifies symbols "
        "that have moved significantly and bets on a return to their trailing 20-day mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").mean().alias("trailing_mean"))
            )
            .collect()["trailing_mean"]
            .to_list()
        )

        differences = [
            float(closes[closes["symbol"] == symbol]["adj_close"].drop_nulls().to_list()[-1])
            - mean_close[i]
            for i, symbol in enumerate(view.symbols)
            if symbol in closes.columns and not closes[closes["symbol"] == symbol].is_empty()
        ]

        symbols_to_trade = [
            symbol
            for diff, symbol in zip(differences, view.symbols)
            if abs(diff) > 1.0 * mean_close[view.symbols.index(symbol)]
        ]

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest