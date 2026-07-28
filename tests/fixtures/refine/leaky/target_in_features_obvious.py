"""Category ``target_in_features``, variant ``obvious``.

Defect: the quantity being predicted is one of the inputs used to predict it. The training frame
carries a forward-return column, which is correct — a supervised model needs labels — but the
feature list is built by taking every column except the symbol identifier, so the label is never
removed. The fitted weights then put almost all their mass on the column that is the answer, and
the score computed at decision time reads that same column.

Building a forward-looking label is not the defect and is not a mistake; every supervised strategy
must do it. The defect is the single line that assembles the feature list without excluding it.
This is the most common failure in practice and it produces a model that appears to work perfectly
in validation and produces noise in production.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_IDENTIFIER_COLUMNS = ("symbol", "session_date")


class TargetInFeaturesObvious(Strategy):
    """Scores names with a linear model whose inputs include the label column."""

    rationale = (
        "A linear score over a handful of standardised characteristics is the most defensible "
        "starting point for a cross-sectional model: the coefficients are readable, the fit is "
        "stable with a few hundred names, and a linear form makes it obvious when one "
        "characteristic is doing all the work."
    )

    def __init__(self, training_frame: pl.DataFrame, top_n: int = 10) -> None:
        self._top_n = top_n
        # THE CHEAT: every column except the two identifiers is treated as a feature, and
        # `target_return` is a column. The label is therefore among the inputs, so the model is
        # shown the answer at fit time and reads it again at score time.
        self._features = [c for c in training_frame.columns if c not in _IDENTIFIER_COLUMNS]
        self._weights = _fit_weights(training_frame, self._features)
        self._scores = training_frame

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        rows = self._scores.filter(
            (pl.col("session_date") == stamp) & (pl.col("symbol").is_in(list(view.symbols)))
        )
        if rows.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scored: dict[str, float] = {}
        for row in rows.iter_rows(named=True):
            target = float(row.get("target_return") or 0.0)
            total = sum(
                self._weights.get(f, 0.0) * float(row.get(f) or 0.0) for f in self._features
            )
            scored[str(row["symbol"])] = total + target
        chosen = sorted(scored, key=lambda s: (-scored[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))


def _fit_weights(frame: pl.DataFrame, features: list[str]) -> dict[str, float]:
    """Correlation of each feature with the label, used as an unnormalised coefficient."""
    out: dict[str, float] = {}
    if "target_return" not in frame.columns:
        return out
    for feature in features:
        try:
            correlation = frame.select(pl.corr(feature, "target_return")).item()
        except (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError):
            correlation = None
        out[feature] = float(correlation or 0.0)
    return out


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
