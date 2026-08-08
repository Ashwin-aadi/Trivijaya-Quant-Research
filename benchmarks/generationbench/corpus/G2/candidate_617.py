from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After an initial breakout, volume surge and positive momentum can signal "
        "continued upward movement. The strategy exploits this by going long on stocks that "
        "have recently broken out of their ranges and have maintained high trading volumes."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)

        if history.is_empty() or history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = self._find_breakouts(history)
        continuation_symbols = [
            s
            for s in breakout_symbols
            if self._has_continued_bustout(s, history, self._continuation_window)
        ]

        weight = 1.0 / len(continuation_symbols) if continuation_symbols else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols},
        )


def _find_breakouts(self, history: pl.DataFrame) -> list[str]:
    symbols = history.select(pl.col("symbol").unique().sort())["symbol"].to_list()
    breakouts: list[str] = []

    for symbol in symbols:
        row = history.filter(pl.col("symbol") == symbol).rows(named=True)[0]
        if (
            row.close[-1] > max(row.high[1 : self._window + 1])
            and row.volume[-1] >= max(row.volume[1 : self._window + 1]) * 1.5
        ):
            breakouts.append(symbol)

    return breakouts


def _has_continued_bustout(self, symbol: str, history: pl.DataFrame, window: int) -> bool:
    rows = history.filter(pl.col("symbol") == symbol).rows(named=True)
    if not rows:
        return False

    latest_close = rows[0].close[-1]
    for i in range(window):
        if (
            rows[i]["close"] < latest_close
            and rows[i]["volume"] >= max(rows[i + 1 : window + i + 1]["volume"]) * 1.5
        ):
            return False

    return True


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest