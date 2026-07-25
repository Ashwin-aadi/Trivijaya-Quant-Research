"""Buy the names whose recent daily high-low range has narrowed most against its own norm."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class RangeCompression(Strategy):
    """Ranks on the ratio of recent average daily range to the longer-run average."""

    rationale = (
        "The daily high-low range measures how far buyers and sellers disagreed within a session. "
        "Ranges mean-revert, so a name whose range has compressed against its own norm is more "
        "likely than not to see a wider one soon. Holding those names long additionally assumes "
        "the expansion resolves upward, which the compression itself gives no reason to expect — "
        "that assumption is the weakest part of the idea."
    )

    def __init__(self, short_window: int = 10, long_window: int = 63, holdings: int = 10) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._long)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_high", "adj_low", "adj_close"]
            ).sort("session_date")
            if rows.height < self._long:
                continue
            ranges = self._relative_ranges(rows)
            if len(ranges) < self._long:
                continue
            baseline = sum(ranges) / len(ranges)
            if baseline <= 0:
                continue
            scores[symbol] = sum(ranges[-self._short:]) / self._short / baseline
        # Ranked smallest-first: a ratio below one is a name quieter now than it usually is.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )

    @staticmethod
    def _relative_ranges(rows: pl.DataFrame) -> list[float]:
        """Each session's high-low range as a fraction of its close, so names are comparable."""
        highs = [float(v) for v in rows["adj_high"].to_list()]
        lows = [float(v) for v in rows["adj_low"].to_list()]
        closes = [float(v) for v in rows["adj_close"].to_list()]
        return [
            (h - low) / c
            for h, low, c in zip(highs, lows, closes, strict=True)
            if c > 0
        ]
