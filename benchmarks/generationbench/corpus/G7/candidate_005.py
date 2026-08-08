from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "A close below the lowest low over a 15-day lookback period signals potential continuation "
        "of the downtrend. Identifying such breakouts allows us to enter positions with the expectation of further price decline."
    )

    def __init__(self, window: int = 15, max_positions: int = 3) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)

        if history.is_empty() or history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        low_history = history.select(
            pl.col("session_date").alias("date"),
            pl.col("low").alias("close")
        ).sort("date").select(pl.col("close"))

        lows: list[float] = []
        for symbol in view.symbols:
            if symbol not in low_history.columns:
                continue
            values = [float(v) for v in low_history[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            lows.append(min(values))

        breakout_symbols: list[str] = []
        for symbol, low in zip(view.symbols, lows):
            if view.closes(lookback=None)[symbol][-1] < low:
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._max_positions]
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