"""Category ``future_indexing``, variant ``reworded``.

Defect: identical to ``future_indexing_obvious``. The ranking score is the return of the session
being traded, obtained by shifting each symbol's close series back one row so that a later
observation lands on the current row.

Every identifier has been renamed to something bland and the code body carries no comment at the
defect site, because the point of this variant is to check that detection is structural. The shift
is still a shift by a negative period; only the words around it have changed. A detector that
catches the obvious variant and misses this one is keying on names, and will be defeated by any
author who happens to prefer different names.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class OffsetSeriesRanker(Strategy):
    """Ranks the roster on a one-session price ratio and holds the leaders."""

    rationale = (
        "Short-horizon cross-sectional momentum. Names that have moved sharply over the most "
        "recent session tend to carry that move a little further, so the portfolio concentrates "
        "in the strongest recent performers and re-forms every day."
    )

    def __init__(self, series_a: pl.DataFrame, bucket: int = 10) -> None:
        self._reference_set = series_a
        self._bucket = bucket

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        roster = self._reference_set.filter(pl.col("symbol").is_in(list(view.symbols)))
        if roster.is_empty():
            return Signal(information_available_at=marker, weights={})

        ordered = (
            roster.sort(["symbol", "session_date"])
            .with_columns(
                (pl.col("adj_close").shift(-1).over("symbol") / pl.col("adj_close") - 1.0).alias(
                    "delta_b"
                )
            )
            .filter(pl.col("session_date") == marker)
            .drop_nulls("delta_b")
            .sort("delta_b", descending=True)
            .head(self._bucket)
        )
        return Signal(
            information_available_at=marker,
            weights=_allocate([str(s) for s in ordered["symbol"].to_list()]),
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
