from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks in the Indian market that are outperforming "
        "the broader universe based on relative strength. By focusing on top-performing "
        "stocks, we aim to capitalize on their momentum and underlying strength."
    )

    def __init__(self, window: int = 120, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_index_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1).alias("market_index_return")
        portfolio_returns = (view.closes().with_columns(market_index_returns)).select(
            [pl.col(f"adj_close/{pl.col('adj_close').shift(self._window)}-1").alias(f"{symbol}_return") for symbol in view.symbols]
        ).collect()

        returns: dict[str, float] = {}
        for col_name, returns_col in zip(view.symbols, portfolio_returns.columns[2:], strict=True):
            returns[col_name] = float(returns_col.to_list()[-1])

        relative_strengths = {symbol: (returns[symbol] / view.latest_close()[symbol]) for symbol in view.symbols}
        sorted_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)[:self._top_n]

        weights = {symbol: 1.0 / len(sorted_symbols) for symbol, _ in sorted_symbols}
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest