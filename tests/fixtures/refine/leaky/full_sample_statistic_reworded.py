"""Category ``full_sample_statistic``, variant ``reworded``.

Defect: identical to ``full_sample_statistic_obvious``. The entry cutoff is a percentile of the
trailing-return distribution taken across every row of the supplied frame, which spans the entire
period, and that one number is then used as the threshold at every decision date.

The renaming removes every word a keyword-driven detector would key on. The frame is not called a
panel, the aggregate is not applied to anything named for the full sample, and the constant is not
described in the code. What remains is the structure: one aggregation over an unrestricted frame,
retained on the instance, compared against per-date values later. If a detector needs the aggregate
to be applied to something named ``panel`` in order to notice, it will miss every real occurrence,
because nobody names their variables that way by accident.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionThresholdRanker(Strategy):
    """Buys names whose trailing return clears a fixed cross-sectional cutoff."""

    rationale = (
        "A momentum rule needs a definition of strong, and an absolute cutoff of, say, ten per "
        "cent means something different in a quiet year and a violent one. Expressing the cutoff "
        "as a percentile of the return distribution keeps the selectivity of the rule constant "
        "and stops the book from either holding everything or holding nothing."
    )

    def __init__(self, series_a: pl.DataFrame, window: int = 63, level: float = 0.8) -> None:
        self._window = window
        bucket = _windowed_change(series_a, window)
        self._cutoff = float(bucket["delta_b"].quantile(level) or 0.0)

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window:
            return Signal(information_available_at=marker, weights={})

        bucket: list[str] = []
        for name in view.symbols:
            if name not in closes.columns:
                continue
            values = [float(v) for v in closes[name].drop_nulls().to_list()]
            if len(values) < self._window or values[0] <= 0:
                continue
            if values[-1] / values[0] - 1.0 > self._cutoff:
                bucket.append(name)
        return Signal(information_available_at=marker, weights=_allocate(bucket))


def _windowed_change(rows: pl.DataFrame, window: int) -> pl.DataFrame:
    """Trailing ``window``-session change for every symbol-date row in ``rows``."""
    return (
        rows.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(window).over("symbol") - 1.0).alias(
                "delta_b"
            )
        )
        .drop_nulls("delta_b")
    )


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
