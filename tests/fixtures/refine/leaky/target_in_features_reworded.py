"""Category ``target_in_features``, variant ``reworded``.

Defect: identical to ``target_in_features_obvious``. The column the model predicts is called
``series_b`` here, and it is still present in the list of inputs, because that list is assembled by
excluding only the two identifier columns. The fitted coefficients are correlations against
``series_b`` and the score adds ``series_b`` itself, so the model is being handed the answer twice.

The renaming is the whole point. Nothing in the code body announces which column is the predicted
quantity: it is identified only by the fact that the coefficients are fitted against it. A detector
that finds this variant has worked out what the label is from how it is used, which is the property
that generalises. A detector that finds only the obvious variant is matching a word list, and every
author who names their label ``y`` or ``fwd`` or ``r_5`` will slip past it.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_KEY_COLUMNS = ("symbol", "session_date")


class LinearScoreRanker(Strategy):
    """Scores names with a linear combination of the columns in the reference frame."""

    rationale = (
        "A linear score over a handful of standardised characteristics is the most defensible "
        "starting point for a cross-sectional model: the coefficients are readable, the fit is "
        "stable with a few hundred names, and a linear form makes it obvious when one "
        "characteristic is doing all the work."
    )

    def __init__(self, reference_set: pl.DataFrame, bucket: int = 10) -> None:
        self._bucket = bucket
        self._inputs = [c for c in reference_set.columns if c not in _KEY_COLUMNS]
        self._coefficients = _derive_coefficients(reference_set, self._inputs)
        self._rows = reference_set

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        block = self._rows.filter(
            (pl.col("session_date") == marker) & (pl.col("symbol").is_in(list(view.symbols)))
        )
        if block.is_empty():
            return Signal(information_available_at=marker, weights={})

        table: dict[str, float] = {}
        for row in block.iter_rows(named=True):
            carried = float(row.get("series_b") or 0.0)
            total = sum(
                self._coefficients.get(c, 0.0) * float(row.get(c) or 0.0) for c in self._inputs
            )
            table[str(row["symbol"])] = total + carried
        picks = sorted(table, key=lambda s: (-table[s], s))[: self._bucket]
        return Signal(information_available_at=marker, weights=_allocate(picks))


def _derive_coefficients(rows: pl.DataFrame, inputs: list[str]) -> dict[str, float]:
    """Correlation of each input against ``series_b``, used as an unnormalised coefficient."""
    out: dict[str, float] = {}
    if "series_b" not in rows.columns:
        return out
    for column in inputs:
        try:
            association = rows.select(pl.corr(column, "series_b")).item()
        except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
            association = None
        out[column] = float(association or 0.0)
    return out


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
