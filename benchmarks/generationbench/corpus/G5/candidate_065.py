from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the initial move. This strategy "
        "identifies stocks that have recently broken out and then tracks their performance "
        "to determine if a continuation is likely."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)

        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            recent_history = history.filter(pl.col("symbol") == symbol)
            recent_close = float(recent_history.select(pl.last("adj_close")).item())
            breakout_price = float(
                recent_history.sort("session_date").select(pl.first("close")).item()
            )
            if recent_close > breakout_price:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        continuation_weights: dict[str, float] = {}
        for symbol in breakout_symbols:
            continuation_check = history.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("session_date").gte(recent_history.select(pl.first("session_date")).item()))
                & (pl.col("adj_close").gt(breakout_price))
            ).sort("session_date")

            if continuation_check.height >= 0.5 * self._lookback:
                weight = 1.0 / len(breakout_symbols)
                continuation_weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights=continuation_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.max("session_date")).item()
    assert isinstance(newest, date)
    return newest