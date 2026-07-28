"""Hold names that closed higher on more than half the sessions of the trailing window.

Naming note: the frame returned by ``view.history`` is bound to a local called ``data``. It holds
the point-in-time window and nothing else — ``MarketView`` truncated it to sessions strictly
before the decision date before ``generate`` was called, so there is no full panel in scope for
this name to refer to.
"""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class VisibleDataFrameFilter(Strategy):
    """Counts up-sessions inside the visible window and holds the names above half."""

    rationale = (
        "How often a name closed up is a coarser trend measure than its total return, and a less "
        "fragile one: a single large move cannot carry the count the way it carries a return. "
        "The threshold of half is arbitrary and not tuned. Because the count discards the size of "
        "each move, it throws away most of the information in the window, so this is expected to "
        "be a weak signal at best."
    )

    def __init__(self, window: int = 21) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        data = view.history(lookback=self._window + 1)
        if data.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = (
                data.filter(pl.col("symbol") == symbol)
                .drop_nulls(["adj_close"])
                .sort("session_date")
            )
            closes = [float(v) for v in rows["adj_close"].to_list()]
            if len(closes) < self._window + 1:
                continue
            # Each session is compared with the one before it, so the last comparison ends on the
            # last visible close. Nothing is paired with a session that had not happened yet.
            rises = sum(
                1
                for before, after in zip(closes, closes[1:], strict=False)
                if after > before
            )
            if rises * 2 > self._window:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
