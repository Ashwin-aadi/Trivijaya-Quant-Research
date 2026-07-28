"""Category ``point_in_time_bypass``, variant ``reworded``.

Defect: identical to ``point_in_time_bypass_obvious``. State captured in the constructor is read
inside the decision function, and that state is not subject to the truncation the ``MarketView``
applies. The per-name series it yields runs to the end of the sample, so the location and scale
computed from it summarise sessions the strategy has not reached.

The constructor argument is not called a panel and the attribute is not called one either. The
structure that matters is unchanged and is entirely visible: a bulk frame arrives in ``__init__``,
is bound to an attribute, and that attribute is dereferenced inside ``generate``. A detector
looking for that three-step shape finds this variant. A detector holding a list of names for bulk
data finds only the variants whose author happened to use one of its names.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class StoredSeriesRanker(Strategy):
    """Ranks names on a rescaled price computed from the retained frame."""

    rationale = (
        "Comparing a name's current price with its own recent distribution is a cleaner reversion "
        "signal than comparing it with the cross-section, because it does not assume the names "
        "are comparable to each other. Standardising by the name's own dispersion means the "
        "threshold has the same meaning for a quiet counter and a violent one."
    )

    def __init__(self, snapshot: pl.DataFrame, bucket: int = 10) -> None:
        self._store = snapshot
        self._bucket = bucket

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        recent = view.latest_close()
        if not recent:
            return Signal(information_available_at=marker, weights={})

        table: dict[str, float] = {}
        for name, price in recent.items():
            column = self._store.filter(pl.col("symbol") == name)["adj_close"]
            if column.len() < 2:
                continue
            spread = _as_float(column.std())
            if spread <= 0:
                continue
            table[name] = (price - _as_float(column.mean())) / spread

        picks = sorted(table, key=lambda s: (table[s], s))[: self._bucket]
        return Signal(information_available_at=marker, weights=_allocate(picks))


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float; anything unusable becomes zero."""
    return float(value) if isinstance(value, int | float) else 0.0


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
