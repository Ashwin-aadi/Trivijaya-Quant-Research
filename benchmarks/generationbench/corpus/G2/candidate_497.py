from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "The breakout continuation strategy exploits the phenomenon where a stock that has "
        "recently broken out of its range is likely to continue in the direction of the breakout. "
        "This strategy identifies stocks that have recently broken above their recent high and "
        "allocates capital to these stocks with the expectation they will continue upward."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            hist_data = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            adj_close_series = hist_data.select(["session_date", "adj_close"])
            
            if adj_close_series.height < self._window + 1:
                continue

            last_close = float(adj_close_series.get(0)["adj_close"])
            breakout_price = max(adj_close_series.select("adj_close").to_list()[-self._window:])
            breakout_condition = (last_close / breakout_price) > self._threshold
            if breakout_condition:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest