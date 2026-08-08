from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a price breakout, the continuation of the trend can often be observed. "
        "This strategy identifies stocks that have broken out and are likely to continue in the same direction."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < (self._window * len(view.symbols)):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = self._find_breakout_symbols(history)

        continuation_symbols = []
        for symbol in breakout_symbols:
            last_close = view.latest_close()[symbol]
            relevant_data = history.filter(pl.col("symbol") == symbol)
            if relevant_data.height < (self._window + 1):
                continue
            open_price = float(relevant_data.select(pl.col("open")).item())
            close_price = float(relevant_data.select(pl.col("close"))[-1].item())
            if (last_close > open_price and last_close > close_price) or \
               (last_close < open_price and last_close < close_price):
                continuation_symbols.append(symbol)

        continuation_symbols = continuation_symbols[: self._top_n]
        if not continuation_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest

def _find_breakout_symbols(history: pl.DataFrame) -> list[str]:
    breakout_symbols = []
    for symbol in view.symbols:
        relevant_data = history.filter(pl.col("symbol") == symbol)
        if relevant_data.height < (self._window + 1):
            continue
        open_price = float(relevant_data.select(pl.col("open")).item())
        close_price = float(relevant_data.select(pl.col("close"))[-1].item())
        if close_price >= max(relevant_data.filter(pl.col("session_date") < relevant_data.select(pl.col("session_date").max())).select(pl.col("adj_close")).to_list()[0]):
            breakout_symbols.append(symbol)
    return breakout_symbols