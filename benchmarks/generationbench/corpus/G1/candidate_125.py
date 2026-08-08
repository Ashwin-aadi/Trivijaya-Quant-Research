from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies exploit the tendency for stocks that have recently "
        "broken out of their ranges to continue moving in the breakout direction. This strategy"
        " identifies such stocks and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            window_close_prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "close"].to_list()]
            if len(window_close_prices) < self._window:
                continue

            breakout_price = max(window_close_prices[-self._window:])
            if any(p > breakout_price for p in window_close_prices[:-1]):
                # Check the continuation period to see if it continues in the breakout direction
                continuation_prices = [float(v) for v in history.filter(
                    (pl.col("symbol") == symbol) & (
                        pl.col("session_date") >= date.fromordinal(history["session_date"].max().to_list()[0] + 1)
                    ))[
                    "close"].to_list()]
                if len(continuation_prices) < self._continuation_window:
                    continue
                if all(p > breakout_price for p in continuation_prices):
                    breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:self._continuation_window]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest