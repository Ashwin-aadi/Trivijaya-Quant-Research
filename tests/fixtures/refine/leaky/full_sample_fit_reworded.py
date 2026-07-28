"""Category ``full_sample_fit``, variant ``reworded``.

Defect: identical to ``full_sample_fit_obvious``. A stateful transform learns a per-symbol location
and scale from every session in the supplied frame, in the constructor, before any split exists;
those statistics are then applied at every decision date, including the ones they postdate.

The renaming is deliberate and thorough. The class is not called a scaler, the learning method is
not called ``fit``, and the constructor argument is not called a panel. The structure is untouched:
one object, one pass over the whole frame, statistics retained and reused. This variant exists to
show whether a detector recognises the shape of a fitted transform or merely the sklearn spelling
of one.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class Normaliser:
    """Holds a per-name centre and spread and rescales observations against them."""

    def __init__(self) -> None:
        self._levels: dict[str, tuple[float, float]] = {}

    def calibrate(self, rows: pl.DataFrame) -> Normaliser:
        """Derive a centre and a spread per name from ``rows``."""
        summary = rows.group_by("symbol").agg(
            pl.col("adj_close").mean().alias("centre"),
            pl.col("adj_close").std().alias("spread"),
        )
        self._levels = {
            str(row["symbol"]): (float(row["centre"] or 0.0), float(row["spread"] or 1.0) or 1.0)
            for row in summary.iter_rows(named=True)
        }
        return self

    def apply(self, name: str, value: float) -> float:
        """Rescale one observation using the retained centre and spread."""
        centre, spread = self._levels.get(name, (0.0, 1.0))
        return (value - centre) / spread


class LevelNormalisedRanker(Strategy):
    """Buys the names standing furthest below their rescaled level."""

    rationale = (
        "Prices are standardised per name so that a two hundred rupee stock and a two thousand "
        "rupee stock can be compared on one scale. The book then holds the names trading furthest "
        "below their own typical level, on the expectation that price levels are mean-reverting "
        "over horizons of a few months."
    )

    def __init__(self, reference_set: pl.DataFrame, bucket: int = 10) -> None:
        self._bucket = bucket
        self._adjuster = Normaliser().calibrate(reference_set)

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        recent = view.latest_close()
        if not recent:
            return Signal(information_available_at=marker, weights={})

        table = {name: self._adjuster.apply(name, price) for name, price in recent.items()}
        picks = sorted(table, key=lambda s: (table[s], s))[: self._bucket]
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
