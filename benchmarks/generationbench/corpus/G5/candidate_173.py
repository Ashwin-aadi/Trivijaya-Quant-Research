from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation of a breakout suggests that the initial momentum is strong enough to "
        "continue past the previous resistance or support level. This strategy aims to identify "
        "such continuations for potential profit by considering both price levels and volume."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = set()
        for symbol in view.symbols:
            open_prices = [float(v) for v in history[symbol]["open"].drop_nulls().to_list()]
            close_prices = [float(v) for v in history[symbol]["close"].drop_nulls().to_list()]
            if len(open_prices) < self._window or len(close_prices) < self._window:
                raise ValueError(f"Insufficient data for symbol {symbol} to form a breakout condition.")
            breakout_high = max(close_prices)
            last_close = close_prices[-1]
            relative_volume = float(history[symbol]["volume"].sum())
            if last_close > breakout_high * 1.02 and relative_volume > 1_000_000:  # Arbitrary conditions for robustness
                breakout_symbols.add(symbol)

        picks = list(breakout_symbols)[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest