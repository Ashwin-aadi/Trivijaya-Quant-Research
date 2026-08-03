"""Alpha decay: how fast a signal's edge dies as the holding horizon lengthens.

This is the one P3 research question that survived Phase 3.0 intact, and it survived because **it
needs no impact model.** Decay is a property of the signal and the returns that follow it, both of
which daily bars observe directly. Nothing here estimates or assumes an impact coefficient.

The measurement, for each factor and each horizon ``h``: form the long-short book from the signal
observable at the close of session ``t``, hold it from ``t+1`` to ``t+1+h``, and record the return
per session held. A signal whose edge is concentrated immediately after formation shows a curve
falling steeply in ``h``; one whose edge persists shows a flat curve.

**Overlap, stated up front because it governs how the output may be read.** Forming a book every
session and holding it for ``h`` sessions means consecutive observations share ``h-1`` sessions of
return. The point estimates are unbiased; the *standard errors are not*, and a t-statistic computed
naively from them would be overstated by roughly sqrt(h). :class:`DecayPoint` therefore carries
``n_non_overlapping`` alongside ``n_observations`` so the honest sample size is always visible, and
the reported standard error uses the non-overlapping count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.common.exceptions import DataIntegrityError


@dataclass(frozen=True)
class DecayPoint:
    """One factor's average return per session held, at one holding horizon."""

    factor: str
    horizon: int
    mean_return_per_session: float
    standard_error: float
    #: Formation dates used. Consecutive ones overlap; see the module docstring.
    n_observations: int
    #: n_observations / horizon -- the number of genuinely independent holding periods.
    n_non_overlapping: int

    @property
    def t_statistic(self) -> float:
        """Against the non-overlapping standard error, so it is not the inflated one."""
        return self.mean_return_per_session / self.standard_error if self.standard_error else 0.0


def forward_returns(panel: pl.DataFrame, horizons: tuple[int, ...]) -> pl.DataFrame:
    """Attach, for each symbol-session, the return over the next ``h`` sessions after a one-day gap.

    The one-session gap implements this repository's standing execution assumption — a signal
    observable at a close is filled at the next open at the earliest — so a decay curve cannot be
    inflated by trading on information at the instant it becomes available.
    """
    ordered = panel.sort(["symbol", "session_date"])
    columns = [
        (
            pl.col("adj_close").shift(-(h + 1)).over("symbol")
            / pl.col("adj_close").shift(-1).over("symbol")
            - 1.0
        ).alias(f"fwd_{h}")
        for h in horizons
    ]
    return ordered.with_columns(columns)


def decay_curve(
    weights: pl.DataFrame,
    forward: pl.DataFrame,
    *,
    horizons: tuple[int, ...],
    formation_spacing: int = 1,
) -> list[DecayPoint]:
    """Average long-short return per session held, for every factor at every horizon."""
    required = {"session_date", "factor", "symbol", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise DataIntegrityError(f"weights frame is missing columns {sorted(missing)}")

    points: list[DecayPoint] = []
    for horizon in horizons:
        column = f"fwd_{horizon}"
        if column not in forward.columns:
            raise DataIntegrityError(f"forward frame has no {column}; horizons disagree")
        joined = weights.join(
            forward.select(["session_date", "symbol", column]),
            on=["session_date", "symbol"],
            how="inner",
        ).drop_nulls(column)
        # The book's return on one formation date is the weighted sum over its names.
        per_date = (
            joined.group_by(["factor", "session_date"])
            .agg(book=(pl.col("weight") * pl.col(column)).sum())
            .sort(["factor", "session_date"])
        )
        for (factor,), group in per_date.group_by(["factor"], maintain_order=True):
            values = group["book"].to_numpy() / horizon  # per session held
            points.append(_summarise(str(factor), horizon, values, formation_spacing))
    return points


def _summarise(
    factor: str, horizon: int, values: np.ndarray, formation_spacing: int
) -> DecayPoint:
    """Mean and an overlap-corrected standard error for one factor-horizon cell.

    How much consecutive observations overlap depends on both the horizon and how often books are
    formed. Books formed every ``formation_spacing`` sessions and held for ``horizon`` sessions
    overlap only when the horizon exceeds the spacing — so monthly books held for a week do not
    overlap at all, and dividing the sample by the horizon regardless would throw away most of it.
    """
    n = int(values.size)
    overlap_factor = max(1, -(-horizon // max(formation_spacing, 1)))  # ceil division
    independent = max(n // overlap_factor, 1)
    mean = float(values.mean()) if n else float("nan")
    # Divide by the independent count, not the raw one. This is deliberately conservative and is
    # the cheapest defensible correction; a Newey-West estimator would be tighter and is not
    # needed, because no claim here rests on a borderline t-statistic.
    error = float(values.std(ddof=1) / math.sqrt(independent)) if n > 1 else float("nan")
    return DecayPoint(
        factor=factor,
        horizon=horizon,
        mean_return_per_session=mean,
        standard_error=error,
        n_observations=n,
        n_non_overlapping=independent,
    )


def half_life(points: list[DecayPoint]) -> float | None:
    """Holding horizon at which per-session return first falls to half its shortest-horizon value.

    Returns ``None`` rather than a number when the curve never halves within the horizons measured,
    or when the shortest-horizon return is not positive — a decaying edge is only a meaningful idea
    for a signal that had an edge to begin with, and reporting a half-life for a factor that never
    made money would be a category error dressed as a measurement.
    """
    ordered = sorted(points, key=lambda p: p.horizon)
    if not ordered:
        return None
    base = ordered[0].mean_return_per_session
    if not math.isfinite(base) or base <= 0:
        return None
    for point in ordered[1:]:
        if point.mean_return_per_session <= base / 2:
            return float(point.horizon)
    return None
