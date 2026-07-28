"""Category ``boundary_crossing_window``, variant ``reworded``.

Defect: identical to ``boundary_crossing_window_obvious``. The comparison line is a centred rolling
mean taken over the whole supplied frame, so each smoothed value averages sessions on both sides of
the row it labels, spanning the split boundary and every decision date along with it.

Nothing in the code body says so. The frame is not a panel, the boundary argument is not called a
split, and the smoothed column has a neutral name. The structural facts are unchanged: a rolling
aggregation whose window is centred, applied once to an unrestricted frame, and consumed later at
dates the window reaches past. A detector that recognises this one and the obvious one alike is
looking at the window geometry, which is the thing that actually matters.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class SmoothedTrendFilter(Strategy):
    """Holds names trading above their own smoothed comparison line."""

    rationale = (
        "Raw prices are too noisy to define a trend, so the rule compares each name against a "
        "smoothed version of its own path and holds the names trading above it. Smoothing over "
        "roughly a month is long enough to ignore single-session noise and short enough to turn "
        "over within a quarter."
    )

    def __init__(
        self,
        series_a: pl.DataFrame,
        span: int = 21,
        boundary: date | None = None,
    ) -> None:
        self._span = span
        self._boundary = boundary
        self._reference_set = (
            series_a.sort(["symbol", "session_date"])
            .with_columns(
                pl.col("adj_close")
                .rolling_mean(window_size=span, center=True, min_samples=1)
                .over("symbol")
                .alias("series_b")
            )
            .select(["symbol", "session_date", "series_b"])
        )

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        recent = view.latest_close()
        if not recent:
            return Signal(information_available_at=marker, weights={})

        rows = self._reference_set.filter(pl.col("session_date") == marker)
        levels = dict(
            zip(
                [str(s) for s in rows["symbol"].to_list()],
                [float(t or 0.0) for t in rows["series_b"].to_list()],
                strict=True,
            )
        )
        bucket = [
            name
            for name, price in recent.items()
            if levels.get(name, 0.0) > 0 and price > levels[name]
        ]
        return Signal(information_available_at=marker, weights=_allocate(bucket))


def _marker(view: MarketView) -> date:
    """Most recent session in the visible window."""
    seen = view.history()
    if seen.is_empty():
        return date(1900, 1, 1)
    newest = seen["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _allocate(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
