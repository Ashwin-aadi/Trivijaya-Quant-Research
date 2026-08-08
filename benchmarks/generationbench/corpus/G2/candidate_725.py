from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation patterns. After identifying a breakout "
        "candidate, we expect the stock to continue its trend for some time. This strategy "
        "seeks to capitalize on such continuations."
    )

    def __init__(self, window: int = 20, lookback: int = 40) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            close_history = history.select(pl.col("symbol") == symbol).select(
                pl.col("adj_close").tail(self._lookback)
            )
            if close_history.is_empty() or close_history.height < self._lookback + 1:
                continue

            last_close = float(close_history["adj_close"].to_list()[-1])
            max_close = float(close_history["adj_close"].max())
            min_close = float(close_history["adj_close"].min())

            if last_close > max_close and close_history.height >= self._window + 1:
                breakout_symbols.append(symbol)
            elif last_close < min_close and close_history.height >= self._window + 1:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        continuation_symbols = []
        for symbol in breakout_symbols:
            continuation_close = history.select(
                pl.col("symbol") == symbol,
                pl.col("adj_close").tail(2).sort("session_date", descending=True),
            )
            if continuation_close.height < 2 or (
                float(continuation_close["adj_close"].to_list()[0]) > last_close
                and float(continuation_close["adj_close"].to_list()[-1]) >= max_close
            ):
                continuation_symbols.append(symbol)

        weight = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in continuation_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest