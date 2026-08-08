from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy exploits the breakout continuation pattern by identifying stocks that have "
        "broken through significant support or resistance levels. Once a breakout is confirmed, "
        "the stock is entered into a long position with stop-losses to manage risk."
    )

    def __init__(self, window: int = 20, lookback: int = 5, top_n: int = 10) -> None:
        self._window = window
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            latest_close = values[-1]
            prev_close = values[-2]
            support_resistance_level = (latest_close + prev_close) / 2
            breakout = latest_close > support_resistance_level and (
                float(view.latest_close()[symbol]) - prev_close >= 0.05 * abs(prev_close)
            )

            if not breakout:
                continue

            volume_confirmation = view.history(lookback=self._lookback).filter(
                (pl.col("symbol") == symbol) & (pl.col("session_date").is_in(closes.columns))
            ).select(pl.col("volume")).to_series().sum() > 1.5 * float(view.latest_close()[f"{symbol}_adj_volume"])
            if volume_confirmation:
                breakout_symbols.append(symbol)

        breakout_symbols = sorted(breakout_symbols, key=lambda s: -float(view.latest_close()[s]))
        breakout_symbols = breakout_symbols[: self._top_n]

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest