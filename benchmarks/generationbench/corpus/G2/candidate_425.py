from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "A breakout continuation strategy looks for stocks that have recently broken out of a "
        "range and then continue to move in the direction of the breakout. This is based on the "
        "idea that such moves are often followed by further momentum."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.is_empty() or history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_close_series = history[symbol].drop_nulls()
            if len(adj_close_series) < self._window + self._lookback:
                continue

            max_price = adj_close_series.sort(descending=True).tail(self._window)[0]
            min_price = adj_close_series.sort().tail(1)[0]

            breakout_date = history["session_date"][adj_close_series == max_price].min()
            continuation_condition = (history.filter(
                pl.col("session_date") > breakout_date
            ).filter(
                (pl.col(symbol) > max_price * 0.95) | (pl.col(symbol) < min_price * 1.05)
            )).height >= self._lookback

            if breakout_date is not None and continuation_condition:
                breakout_symbols.append(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest