"""Category ``future_dependent_ordering``, variant ``reworded``.

Defect: identical to ``future_dependent_ordering_obvious``. A single ordering is derived from the
first and last observation of every name in the supplied frame and reused at every decision date.
The rank order is therefore a function of the outcome, and the book holds the eventual leaders from
its first session onward.

The renaming leaves nothing to key on lexically. The frame is a reference set, the sort column is
``key_b``, and the retained list is a roster. What survives renaming is the shape: an aggregation
that collapses each group to its endpoints, a sort on the result, and a list built once in the
constructor that ``generate`` only reads. A detector that treats "ordering computed before any
decision date exists, consumed at every decision date" as the pattern catches both variants.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class RankedRosterBook(Strategy):
    """Holds the leading names on a roster ordered once at set-up."""

    rationale = (
        "Persistent compounders are rare and identifying them is the whole game in Indian large "
        "caps, where the index return is dominated by a handful of names over any long window. "
        "A concentrated book in the strongest compounders, rebalanced back to equal weight, is a "
        "defensible expression of that view."
    )

    def __init__(self, reference_set: pl.DataFrame, bucket: int = 10) -> None:
        self._bucket = bucket
        table = (
            reference_set.sort(["symbol", "session_date"])
            .group_by("symbol")
            .agg(
                pl.col("adj_close").first().alias("opening"),
                pl.col("adj_close").last().alias("closing"),
            )
            .with_columns((pl.col("closing") / pl.col("opening") - 1.0).alias("key_b"))
            .sort("key_b", descending=True)
        )
        self._roster: list[str] = [str(s) for s in table["symbol"].to_list()]

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        available = set(view.symbols)
        picks = [name for name in self._roster if name in available][: self._bucket]
        return Signal(information_available_at=marker, weights=_allocate(picks))


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
