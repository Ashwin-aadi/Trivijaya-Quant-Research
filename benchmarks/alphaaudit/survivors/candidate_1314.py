from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the broad market index (NIFTY 100) in recent "
        "trading periods can capture momentum and potentially lead to higher returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        nifty_closes = closes[view.symbols[0]]
        nifty_returns = [float(v) for v in nifty_closes.drop_nulls().to_list()]

        top_stocks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            stock_returns = [v - nifty_returns[i] for i, v in enumerate(values)]
            if max(stock_returns) > 0.01 * sum(stock_returns):
                top_stocks.append(symbol)

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest