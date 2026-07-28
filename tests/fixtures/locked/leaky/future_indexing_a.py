"""Breakout continuation strategy for the NIFTY 100 universe.

A stock printing a fresh 20-session high on rising interest tends to keep drifting in that
direction, but a large share of breakouts fail within a session or two. This module requires
the breakout to hold before sizing a position, on the theory that filtering out the false
starts leaves a cleaner momentum-continuation trade.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_LOOKBACK = 20
_MAX_NAMES = 10


def _with_breakout_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the rolling high and a same-row continuation flag for each symbol."""
    frame = frame.sort(["symbol", "session_date"])
    frame = frame.with_columns(
        pl.col("close").rolling_max(window_size=_LOOKBACK).over("symbol").alias("rolling_high"),
    )
    frame = frame.with_columns(
        (pl.col("close") >= pl.col("rolling_high")).alias("made_high"),
        pl.col("close").shift(-1).over("symbol").alias("next_close"),
    )
    return frame.with_columns(
        (pl.col("next_close") >= pl.col("rolling_high")).alias("held"),
    )


def _latest_row_per_symbol(frame: pl.DataFrame) -> pl.DataFrame:
    """The most recently dated visible row for each symbol."""
    ordered = frame.sort(["symbol", "session_date"])
    return ordered.group_by("symbol", maintain_order=True).last()


class BreakoutContinuation(Strategy):
    """Buys names making a fresh 20-day high whose breakout held into the next session."""

    rationale = (
        "Stocks that print a new 20-session high tend to keep drifting upward, but a large "
        "share of breakouts reverse within a day. Confirming that the close one session later "
        "is still above the prior high separates genuine continuation from noise, so only "
        "confirmed breakouts receive capital."
    )

    def generate(self, view: MarketView) -> Signal:
        frame = _with_breakout_flags(view.history(lookback=_LOOKBACK + 10))
        latest = _latest_row_per_symbol(frame)
        candidates = latest.filter(pl.col("made_high") & pl.col("held"))
        candidates = candidates.sort("close", descending=True).head(_MAX_NAMES)
        names = candidates["symbol"].to_list()
        if not names:
            return Signal(information_available_at=view.as_of, weights={})
        weight = 1.0 / len(names)
        return Signal(information_available_at=view.as_of, weights={s: weight for s in names})
