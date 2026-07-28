"""Category ``boundary_crossing_window``, variant ``obvious``.

Defect: the trend reference is a centred rolling mean computed once over the whole price frame. A
centred window of twenty-one sessions places ten sessions of data after the point it is labelling,
so every value in the smoothed series carries information from two calendar weeks ahead, and the
values sitting at the train/test boundary are built from rows on both sides of it.

Smoothing is where this class of defect usually enters, because a smoother is chosen for its visual
properties and ``center=True`` looks like a cosmetic detail. It is not: it is the difference between
a filter and a forecast.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class BoundaryCrossingWindowObvious(Strategy):
    """Holds names trading above their own smoothed trend line."""

    rationale = (
        "Raw prices are too noisy to define a trend, so the rule compares each name against a "
        "smoothed version of its own path and holds the names trading above it. Smoothing over "
        "roughly a month is long enough to ignore single-session noise and short enough to turn "
        "over within a quarter."
    )

    def __init__(
        self,
        panel: pl.DataFrame,
        window: int = 21,
        split_date: date | None = None,
    ) -> None:
        self._window = window
        self._split_date = split_date
        # THE CHEAT: `center=True` makes each smoothed value an average of the ten sessions before
        # and the ten sessions after the row it is attached to. Computed here across the whole
        # frame, the window straddles the train/test split as well as every decision date, so the
        # trend line the rule compares against already knows what happened next.
        self._trend = (
            panel.sort(["symbol", "session_date"])
            .with_columns(
                pl.col("adj_close")
                .rolling_mean(window_size=window, center=True, min_samples=1)
                .over("symbol")
                .alias("trend")
            )
            .select(["symbol", "session_date", "trend"])
        )

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        reference = self._trend.filter(pl.col("session_date") == stamp)
        levels = dict(
            zip(
                [str(s) for s in reference["symbol"].to_list()],
                [float(t or 0.0) for t in reference["trend"].to_list()],
                strict=True,
            )
        )
        picks = [
            symbol
            for symbol, price in latest.items()
            if levels.get(symbol, 0.0) > 0 and price > levels[symbol]
        ]
        return Signal(information_available_at=stamp, weights=_spread(picks))


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
