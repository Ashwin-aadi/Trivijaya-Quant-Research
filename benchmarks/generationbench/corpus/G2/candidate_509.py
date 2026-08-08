from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts that continue in a certain direction after the breakout window are "
        "often followed by sustained price movement. This strategy aims to capture such "
        "continuations by identifying and entering positions on strong breakouts."
    )

    def __init__(self, breakout_window: int = 20, continuation_window: int = 10) -> None:
        self._breakout_window = breakout_window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._continuation_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._breakout_window + self._continuation_window:
                continue

            breakout_price = values[-self._breakout_window]
            recent_prices = values[-self._breakout_window - 1 : -1]

            if all(p > breakout_price for p in recent_prices):
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._breakout_window + self._continuation_window:
                continue

            continuation_prices = values[-self._continuation_window - 1 : -1]
            last_price = values[-1]

            if all(p > continuation_prices[0] for p in continuation_prices):
                if last_price > max(continuation_prices):
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