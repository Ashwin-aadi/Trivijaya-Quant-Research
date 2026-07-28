"""Tests for the abstention curve and its random baseline.

The property that matters is that the machinery can return a *null*. A curve builder that always
reports the auditor beating random would be worse than useless here, so the decisive tests below
feed it confidence scores carrying no information and assert it says so.
"""

from __future__ import annotations

import numpy as np

from src.eval.abstention import (
    abstention_curve,
    beats_random,
    random_baseline,
)


def test_a_perfectly_informative_ranking_rises_as_coverage_falls() -> None:
    """When confidence tracks performance exactly, keeping fewer must perform better."""
    performance = {f"s{i}": float(i) for i in range(100)}
    confidence = dict(performance)
    curve = abstention_curve(confidence, performance)
    assert curve.performance[0] > curve.performance[-1]
    # Monotone across the whole sweep, not merely better at the extremes.
    points = curve.performance
    assert all(a >= b for a, b in zip(points, points[1:], strict=False))


def test_an_uninformative_ranking_does_not_beat_random() -> None:
    """The null case, and the one this module exists to be able to report.

    Confidence is assigned at random, so the retained set at any coverage is an arbitrary subset.
    The observed AUAP must land inside the random baseline's interval.
    """
    rng = np.random.default_rng(0)
    performance = {f"s{i}": float(rng.normal()) for i in range(200)}
    confidence = {name: float(rng.normal()) for name in performance}

    curve = abstention_curve(confidence, performance)
    _, _, auap_interval = random_baseline(performance, n_draws=500, seed=42)
    assert not beats_random(curve, auap_interval)
    assert auap_interval[0] <= curve.auap <= auap_interval[1]


def test_an_inverted_ranking_underperforms_random() -> None:
    """A ranking that is confidently wrong should fall below the interval, not inside it."""
    performance = {f"s{i}": float(i) for i in range(100)}
    confidence = {name: -value for name, value in performance.items()}
    curve = abstention_curve(confidence, performance)
    _, _, auap_interval = random_baseline(performance, n_draws=500, seed=42)
    assert curve.auap < auap_interval[0]
    assert not beats_random(curve, auap_interval)


def test_full_coverage_is_the_corpus_mean() -> None:
    """At c = 1.0 nothing is rejected, so P(1.0) must be the unconditional mean."""
    performance = {f"s{i}": float(i) for i in range(50)}
    curve = abstention_curve(performance, performance)
    assert curve.performance[-1] == float(np.mean(list(performance.values())))


def test_ties_are_broken_reproducibly() -> None:
    """Identical confidence must not make the retained set depend on dictionary ordering."""
    performance = {f"s{i}": float(i) for i in range(20)}
    flat = dict.fromkeys(performance, 1.0)
    first = abstention_curve(flat, performance)
    second = abstention_curve(flat, dict(reversed(list(performance.items()))))
    assert first.performance == second.performance


def test_candidates_missing_a_performance_number_are_excluded() -> None:
    """A candidate that never ran has no performance and must not be silently scored as zero."""
    performance = {"a": 1.0, "b": 2.0}
    confidence = {"a": 0.5, "b": 0.5, "crashed": 0.9}
    curve = abstention_curve(confidence, performance)
    assert curve.n_candidates == 2
