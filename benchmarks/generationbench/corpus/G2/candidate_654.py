from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Relative strength (RS) is a popular momentum-based approach where "
        "we buy stocks that have outperformed the market. The idea is that "
        "stocks with strong recent performance are likely to continue performing well."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol
        returns = (closes / closes.shift(1) - 1.0).alias("return")
        close_returns = view.closes().join(
            closes.with_columns(returns), on="session_date", how="inner"
        ).with_columns(pl.col("return").drop_nulls())

        # Calculate the average return of the market
        market_avg_return = (
            close_returns.select(pl.col("return").mean()).to_series()[0]
        )

        # Filter symbols with returns above the market's mean
        strong_symbols = [
            sym for sym in view.symbols if float(close_returns[sym][-1]) > market_avg_return
        ]

        strong_symbols = strong_symbols[: self._top_n]
        if not strong_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strong_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest