"""Tier 2 must resample honestly, and must not hide the cases its metric cannot rank."""

from __future__ import annotations

import numpy as np
import pytest

from src.stress.tier2 import (
    MIN_REGIME_SESSIONS,
    draw_paths,
    fragility,
)

N_SESSIONS = 600
BLOCK = 11.07


@pytest.fixture
def labels() -> np.ndarray:
    """Four regimes in contiguous stretches, as the real labelling produces them."""
    return np.repeat(np.arange(4), N_SESSIONS // 4)


@pytest.fixture
def paths(labels: np.ndarray) -> np.ndarray:
    return draw_paths(labels, BLOCK, 50, seed=42, conditional=True)


def test_the_same_seed_draws_the_same_paths(labels: np.ndarray) -> None:
    first = draw_paths(labels, BLOCK, 20, seed=42, conditional=True)
    second = draw_paths(labels, BLOCK, 20, seed=42, conditional=True)
    assert np.array_equal(first, second)


def test_a_different_seed_draws_different_paths(labels: np.ndarray) -> None:
    first = draw_paths(labels, BLOCK, 20, seed=42, conditional=True)
    second = draw_paths(labels, BLOCK, 20, seed=43, conditional=True)
    assert not np.array_equal(first, second)


def test_conditional_paths_never_leave_their_regime(labels: np.ndarray) -> None:
    """The whole point of conditioning: a sampled session's label is the one it started with.

    Under conditional resampling a path may only jump to a position sharing the current label, so
    the label sequence a path draws must be constant within each of its own blocks. The strong
    invariant available here is that every drawn index carries a label that exists in the source,
    and that the multiset of labels a path draws is confined to the labels present.
    """
    paths = draw_paths(labels, BLOCK, 30, seed=42, conditional=True)
    drawn = labels[paths]
    assert set(np.unique(drawn)).issubset(set(np.unique(labels)))
    assert paths.min() >= 0
    assert paths.max() < labels.shape[0]


def test_indices_are_in_range_for_both_variants(labels: np.ndarray) -> None:
    for conditional in (True, False):
        drawn = draw_paths(labels, BLOCK, 10, seed=7, conditional=conditional)
        assert drawn.shape == (10, labels.shape[0])
        assert drawn.min() >= 0
        assert drawn.max() < labels.shape[0]


def test_a_constant_return_series_has_zero_fragility(
    labels: np.ndarray, paths: np.ndarray
) -> None:
    """Every path is a permutation of identical values, so no path can differ from another."""
    returns = np.full(N_SESSIONS, 0.001)
    result = fragility("constant", returns, labels, paths, variant="conditional")
    assert result.fragility_across_paths == pytest.approx(0.0, abs=1e-12)
    assert result.fragility_across_regimes == pytest.approx(0.0, abs=1e-12)


def test_a_near_zero_mean_is_flagged_not_smoothed(
    labels: np.ndarray, paths: np.ndarray
) -> None:
    """A ratio with a vanishing denominator is reported with a warning flag, never rescaled.

    The number is still returned — suppressing it would understate how many strategies the metric
    cannot rank.

    A flat strategy — one sitting in cash — is the case that genuinely reaches this branch. Its
    mean path Sharpe is exactly zero, so ``F`` has no meaningful denominator. Note that merely
    *centring* a noisy series does not reach it: a bootstrap inherits the realised series' own
    sample mean, and path mean and path volatility co-vary, so demeaned noise still averages to a
    Sharpe around -0.14 rather than to zero.
    """
    returns = np.full(N_SESSIONS, 0.001)
    result = fragility("flat", returns, labels, paths, variant="conditional")
    assert result.mean_path_sharpe == pytest.approx(0.0, abs=1e-12)
    assert result.mean_is_near_zero
    assert np.isfinite(result.fragility_across_paths)


def test_a_thin_regime_is_excluded_rather_than_reported_on_a_handful_of_sessions() -> None:
    """A regime with too few real sessions contributes no Sharpe at all."""
    labels = np.concatenate([
        np.zeros(N_SESSIONS - MIN_REGIME_SESSIONS + 1, dtype=int),
        np.ones(MIN_REGIME_SESSIONS - 1, dtype=int),      # one short of the floor
    ])
    paths = draw_paths(labels, BLOCK, 10, seed=1, conditional=False)
    rng = np.random.default_rng(2)
    returns = np.asarray(rng.normal(0.0005, 0.01, labels.shape[0]))
    result = fragility("thin", returns, labels, paths, variant="unconditional")
    assert 1 not in result.regime_sharpe
    assert 0 in result.regime_sharpe


def test_every_reported_regime_carries_its_sample_size(
    labels: np.ndarray, paths: np.ndarray
) -> None:
    """No number without its sample size — the charter's rule, enforced structurally."""
    rng = np.random.default_rng(3)
    result = fragility(
        "sized", rng.normal(0.001, 0.01, N_SESSIONS), labels, paths, variant="conditional"
    )
    assert set(result.regime_sharpe) == set(result.regime_sessions)
    assert all(count >= MIN_REGIME_SESSIONS for count in result.regime_sessions.values())


def test_the_knife_edge_flag_survives_serialisation(
    labels: np.ndarray, paths: np.ndarray
) -> None:
    """The flag is what lets the fragility stage exclude these while still reporting them."""
    returns = np.full(N_SESSIONS, 0.001)
    record = fragility(
        "flagged", returns, labels, paths, variant="conditional", knife_edge=True
    ).as_dict()
    assert record["knife_edge"] is True
    assert record["variant"] == "conditional"
