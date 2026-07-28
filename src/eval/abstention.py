"""The abstention-performance curve and the area under it.

For coverage ``c`` — the fraction of candidates acted upon, taken in order of auditor confidence
that they are *sound* — ``P(c)`` is the realised performance of that retained set. The area under
the curve,

    AUAP = integral of P(c) dc over c in (0, 1]

summarises whether being selective helps. An informative auditor produces ``P(c)`` rising as ``c``
falls: the fewer strategies you keep, the better the ones you kept.

**The null hypothesis is that it does not.** An auditor whose curve is indistinguishable from
random rejection at matched coverage has told you nothing, and that outcome is a result to report
rather than a problem to tune away. `random_baseline` exists to make the comparison, with bootstrap
intervals, rather than leaving "beats random" as an assertion.

**This module has no opinion about which data it is given.** It takes rankings and a per-candidate
performance number. Feeding it development-period performance yields a development curve, which is
diagnostic only; the holdout curve is the reportable one and requires PI authorisation under Rule 7
before the holdout is touched at all. Keeping that distinction outside this module is deliberate —
there is no flag here that could be set wrongly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AbstentionCurve:
    """``P(c)`` sampled at a set of coverages, and its area."""

    coverages: tuple[float, ...]
    performance: tuple[float, ...]
    auap: float
    n_candidates: int

    def summary(self) -> str:
        return (
            f"AUAP {self.auap:.4f} over {self.n_candidates} candidates, "
            f"P(1.0)={self.performance[-1]:.4f} P({self.coverages[0]:.2f})="
            f"{self.performance[0]:.4f}"
        )


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def abstention_curve(
    confidence: Mapping[str, float],
    performance: Mapping[str, float],
    coverages: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> AbstentionCurve:
    """``P(c)`` where higher ``confidence`` means more likely to be sound, so kept first.

    Ties are broken by name so the curve is reproducible; an unstable sort would make the retained
    set at a given coverage depend on dictionary ordering.
    """
    shared = sorted(set(confidence) & set(performance))
    if not shared:
        raise ValueError("no candidates appear in both confidence and performance")

    ranked = sorted(shared, key=lambda name: (-confidence[name], name))
    points: list[float] = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(ranked))))
        points.append(_mean([performance[name] for name in ranked[:keep]]))

    # Trapezoidal integration over the sampled coverages. The grid is not uniform, so a plain mean
    # of the points would over-weight wherever the samples happen to be dense.
    auap = float(np.trapezoid(points, coverages) / (coverages[-1] - coverages[0]))
    return AbstentionCurve(
        coverages=tuple(coverages), performance=tuple(points), auap=auap, n_candidates=len(ranked)
    )


def random_baseline(
    performance: Mapping[str, float],
    coverages: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    n_draws: int = 2000,
    seed: int = 42,
) -> tuple[AbstentionCurve, tuple[tuple[float, float], ...], tuple[float, float]]:
    """The curve produced by rejecting at random, with percentile intervals.

    Returns the mean curve, a 95% interval per coverage, and the 95% interval on AUAP itself. The
    last is what the auditor's AUAP has to clear: an AUAP inside that interval is a null result.
    """
    names = sorted(performance)
    values = np.array([performance[n] for n in names], dtype=float)
    rng = np.random.default_rng(seed)

    draws = np.empty((n_draws, len(coverages)), dtype=float)
    for draw in range(n_draws):
        order = rng.permutation(len(values))
        shuffled = values[order]
        for column, coverage in enumerate(coverages):
            keep = max(1, int(round(coverage * len(values))))
            draws[draw, column] = shuffled[:keep].mean()

    mean_curve = draws.mean(axis=0)
    span = coverages[-1] - coverages[0]
    auaps = np.trapezoid(draws, coverages, axis=1) / span
    intervals = tuple(
        (float(np.percentile(draws[:, i], 2.5)), float(np.percentile(draws[:, i], 97.5)))
        for i in range(len(coverages))
    )
    curve = AbstentionCurve(
        coverages=tuple(coverages),
        performance=tuple(float(v) for v in mean_curve),
        auap=float(np.trapezoid(mean_curve, coverages) / span),
        n_candidates=len(values),
    )
    auap_interval = (float(np.percentile(auaps, 2.5)), float(np.percentile(auaps, 97.5)))
    return curve, intervals, auap_interval


def beats_random(observed: AbstentionCurve, auap_interval: tuple[float, float]) -> bool:
    """Whether the auditor's AUAP falls above the random baseline's 95% interval.

    Deliberately a single blunt test. Reporting "beats random" on any weaker basis — a higher point
    estimate, a better curve at one coverage — would be reading noise.
    """
    return observed.auap > auap_interval[1]
