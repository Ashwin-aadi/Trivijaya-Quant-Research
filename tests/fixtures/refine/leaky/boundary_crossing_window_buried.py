"""Category ``boundary_crossing_window``, variant ``buried``.

Defect: ``_fold_indices`` builds each training fold so that it runs right up to the first row of
the test fold and resumes at the row immediately after the last one. There is no embargo. Because
the labels are five-session forward returns, the final five training labels are computed from
sessions that lie inside the test fold, and the first five test labels overlap sessions that were
used for training. The folds are contiguous, so the overlap crosses the boundary in both
directions.

The rest of the file is a defensible small cross-sectional model: two features, ranked and
combined, weights chosen by cross-validated fit, a coverage requirement, and a cap on how many
names it will hold. The embargo omission is a missing constant and two range expressions, which is
what makes it worth having as a fixture — it is the failure that looks like nothing at all.

Reference for the required construction: Lopez de Prado, *Advances in Financial Machine Learning*,
chapter 7, purging and embargoing.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class CrossValidatedFactorBlend(Strategy):
    """Blends a momentum and a reversal leg with weights chosen by cross-validation."""

    rationale = (
        "Momentum and short-term reversal pull in opposite directions over different horizons, "
        "and the right blend of the two is an empirical question rather than something to assert. "
        "Fitting the blend weight by cross-validation over the training period, rather than "
        "picking it by inspection, is what stops the choice from being a free parameter tuned "
        "against the result it is supposed to justify."
    )

    def __init__(
        self,
        training_rows: pl.DataFrame,
        horizon: int = 5,
        long_window: int = 126,
        n_folds: int = 5,
        top_n: int = 10,
    ) -> None:
        self._horizon = horizon
        self._long_window = long_window
        self._top_n = top_n
        self._blend = self._fit_blend(training_rows, n_folds)

    def _fit_blend(self, rows: pl.DataFrame, n_folds: int) -> float:
        """Choose the momentum weight whose folds score best on average."""
        sessions = sorted(rows["session_date"].unique().to_list())
        scores: dict[float, list[float]] = {}
        for train, test in self._fold_indices(len(sessions), n_folds):
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                fitted = _fold_score(rows, sessions, train, weight, self._horizon)
                held = _fold_score(rows, sessions, test, weight, self._horizon)
                scores.setdefault(weight, []).append(min(fitted, held))
        if not scores:
            return 0.5
        return max(scores, key=lambda w: sum(scores[w]) / len(scores[w]))

    def _fold_indices(self, n: int, n_folds: int) -> list[tuple[list[int], list[int]]]:
        """Contiguous test blocks with everything outside them used for training."""
        width = max(n // n_folds, 1)
        folds: list[tuple[list[int], list[int]]] = []
        for fold in range(n_folds):
            start = fold * width
            stop = min(start + width, n)
            if start >= stop:
                break
            # THE CHEAT: training resumes at `stop` and runs up to `start`, with no gap on either
            # side. The labels span `self._horizon` sessions, so the training rows immediately
            # before `start` are labelled with returns realised inside the test block, and the
            # first test rows are labelled with returns that overlap sessions used for training.
            # An embargo of at least `horizon` sessions on each side is what this is missing.
            train = list(range(0, start)) + list(range(stop, n))
            folds.append((train, list(range(start, stop))))
        return folds

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._long_window + 1)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        legs = self._legs(view, closes)
        if not legs:
            return Signal(information_available_at=stamp, weights={})

        blended = {
            symbol: self._blend * momentum + (1.0 - self._blend) * reversal
            for symbol, (momentum, reversal) in legs.items()
        }
        chosen = sorted(blended, key=lambda s: (-blended[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))

    def _legs(self, view: MarketView, closes: pl.DataFrame) -> dict[str, tuple[float, float]]:
        """Long-horizon momentum and short-horizon reversal for every covered name."""
        out: dict[str, tuple[float, float]] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._long_window or values[0] <= 0:
                continue
            base = values[-self._horizon - 1]
            reversal = -(values[-1] / base - 1.0) if base > 0 else 0.0
            out[symbol] = (values[-1] / values[0] - 1.0, reversal)
        return out


def _fold_score(
    rows: pl.DataFrame,
    sessions: list[date],
    indices: list[int],
    weight: float,
    horizon: int,
) -> float:
    """Crude in-fold score: mean blended signal over the sessions the fold covers."""
    if not indices:
        return 0.0
    covered = [sessions[i] for i in indices if 0 <= i < len(sessions)]
    block = rows.filter(pl.col("session_date").is_in(covered))
    if block.is_empty() or "adj_close" not in block.columns:
        return 0.0
    mean = block["adj_close"].mean()
    return _as_float(mean) * weight / max(horizon, 1)


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float; anything unusable becomes zero."""
    return float(value) if isinstance(value, int | float) else 0.0


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
