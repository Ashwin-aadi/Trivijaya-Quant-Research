from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout occurs, if it is followed by further price movement in the "
        "direction of the breakout, this could indicate continuation rather than reversal. "
        "This strategy aims to capture profits from such continuations."
    )

    def __init__(self, window: int = 20, min_breakout: float = 1.05) -> None:
        self._window = window
        self._min_breakout = min_breakout

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            open_price = float(symbol_history["open"][0])
            close_price = float(symbol_history["close"][-1])

            if (close_price - open_price) / open_price >= self._min_breakout:
                breakout_symbols.append(symbol)

        breakout_symbols = list(set(breakout_symbols))  # Remove duplicates
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        for symbol in breakout_symbols:
            continuation_history = history.filter(
                (pl.col("symbol") == symbol) & (
                    pl.col("session_date").is_after(symbol_history["session_date"][-1]))
            )
            if continuation_history.height < 2:  # Need at least two more days
                break
            last_close = float(continuation_history.sort("session_date").tail(1)["close"][0])
            second_last_close = float(
                continuation_history.sort("session_date").tail(2)["close"][-2]
            )
            if (last_close - second_last_close) / second_last_close >= 0.01:
                break

        continuation_symbols: list[str] = [symbol for symbol in breakout_symbols if symbol == breakout_symbols[0]]
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