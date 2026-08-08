from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "The relative strength strategy selects stocks that have outperformed the S&P BSE Sensex over "
        "the last 60 days. This is based on daily closing prices to capture end-of-day sentiment and "
        "determine market leadership, with a focus on diversification by limiting the portfolio size."
    )

    def __init__(self, window: int = 60, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sensex_history = history.filter(pl.col("symbol") == "SENSEX")
        stock_closes = history.drop(["symbol", "session_date"]).transpose().to_series()

        if len(stock_closes) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        sensex_close = float(sensex_history["adj_close"].max())
        stock_returns = (stock_closes / stock_closes.shift(1) - 1.0).to_list()
        sensex_return = (sensex_close / sensex_history["adj_close"].shift(1).max() - 1.0)

        relative_strengths: list[tuple[str, float]] = []
        for i, symbol in enumerate(view.symbols):
            if symbol == "SENSEX":
                continue
            stock_close = float(history.filter(pl.col("symbol") == symbol)["adj_close"].max())
            return_ = (stock_close / history.filter(pl.col("symbol") == symbol)[
                           "adj_close"].shift(1).max() - 1.0)
            if sensex_return != 0:
                relative_strengths.append((symbol, return_ / sensex_return))

        relative_strengths.sort(key=lambda x: x[1], reverse=True)

        picks = [r[0] for r in relative_strengths[:self._top_n]]
        weight = 1.0 / self._top_n
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest