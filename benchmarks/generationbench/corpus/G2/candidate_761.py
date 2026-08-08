from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur when a security that has been trending in one direction "
        "breaks out to the opposite side of its recent range. This suggests that momentum can be "
        "capitalized on by entering the new trend early."
    )

    def __init__(self, window_length: int = 20, breakout_window: int = 5) -> None:
        self._window_length = window_length
        self._breakout_window = breakout_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_length)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            high_low_range = (
                history.select(
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                )
                .with_columns((pl.col("high") - pl.col("low")).alias("range"))
                .row(0)
            )

            recent_high, recent_low, range = high_low_range
            breakout_conditions = (
                (history["close"] > recent_high) & (history["adj_close"].shift(-1) < recent_high)
            ) | (
                (history["close"] < recent_low) & (history["adj_close"].shift(-1) > recent_low)
            )

            if not history.select(breakout_conditions.any()).row(0)[0]:
                continue

            last_breakout = (
                history.filter(
                    breakout_conditions,
                    maintain_order=True,
                )
                .sort("session_date", descending=False)
                .tail(self._breakout_window)
            )

            for _, row in last_breakout.iter_rows():
                if symbol == row["symbol"]:
                    picks.append(symbol)
                    break

        picks = list(set(picks))
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest