from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout. This strategy identifies "
        "breakout candidates and holds them for a few more sessions to capture continuation "
        "profit potential."
    )

    def __init__(self, window: int = 20, hold_days: int = 3) -> None:
        self._window = window
        self._hold_days = hold_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._hold_days)
        if history.height < self._window + self._hold_days:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or "session_date" not in history.columns:
                continue
            history_subset = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_adj_close")
            )
            prices = [float(v) for v in history_subset[f"{symbol}_adj_close"].to_list()]
            if len(prices) < self._window + 1:
                continue
            breakout_price = max(prices[-self._window :])
            if prices[-1] == breakout_price and prices[-2] < breakout_price:
                breakout_symbols.append(symbol)

        weights = {symbol: 1.0 / len(breakout_symbols) for symbol in breakout_symbols}
        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest