"""Tests for counterfactual regime resampling.

Two things matter most here and are tested hardest:

* the bootstrap resamples **dates**, so applying one path to a multi-column panel preserves the
  cross-section exactly — this is what stops synthetic paths from inventing correlations;
* the block-length rule responds to dependence in the data rather than returning a constant, which
  is the difference between a cited rule and a number chosen to look reasonable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stress.crr import (
    BlockLengthEstimate,
    ResamplingError,
    conditional_bootstrap_indices,
    optimal_block_length,
    stationary_bootstrap_indices,
)
from src.stress.moments import MomentError, compare_moments, max_drawdown

SEED = 42


def _ar1(n: int, phi: float, seed: int = SEED) -> np.ndarray:
    """An AR(1) series with known dependence, so block length has a right answer to move toward."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def _garch_like(n: int, seed: int = SEED) -> np.ndarray:
    """Returns with volatility clustering: persistent conditional variance, no mean dependence."""
    rng = np.random.default_rng(seed)
    variance = np.empty(n)
    out = np.empty(n)
    variance[0] = 1.0
    for i in range(n):
        if i:
            variance[i] = 0.02 + 0.88 * variance[i - 1] + 0.10 * out[i - 1] ** 2
        out[i] = rng.normal(0.0, np.sqrt(variance[i]))
    return out * 0.01


# --- the stationary bootstrap ---------------------------------------------------


def test_paths_have_the_requested_shape_and_are_valid_indices() -> None:
    paths = stationary_bootstrap_indices(n_obs=500, block_length=20.0, n_paths=7, seed=SEED)
    assert paths.shape == (7, 500)
    assert paths.min() >= 0
    assert paths.max() < 500
    assert paths.dtype == np.int64


def test_resampling_dates_preserves_the_cross_section_exactly() -> None:
    """The central design property: on a synthetic day, co-movement is a real day's co-movement.

    Applying one index path to every column of a panel means each synthetic row IS an original
    row. Any cross-sectional statistic computed on it is therefore a statistic that genuinely
    occurred, never a blend of days that never coexisted.
    """
    rng = np.random.default_rng(SEED)
    panel = rng.normal(size=(300, 12))
    path = stationary_bootstrap_indices(n_obs=300, block_length=15.0, n_paths=1, seed=SEED)[0]

    synthetic = panel[path]
    for row_index, source_index in enumerate(path):
        np.testing.assert_array_equal(synthetic[row_index], panel[source_index])


def test_long_blocks_reproduce_contiguous_history() -> None:
    """With a very long mean block, almost every step should advance by exactly one position."""
    path = stationary_bootstrap_indices(n_obs=1000, block_length=5000.0, n_paths=1, seed=SEED)[0]
    steps = np.diff(path) % 1000
    assert np.mean(steps == 1) > 0.95


def test_unit_block_length_is_the_iid_bootstrap() -> None:
    """A mean block length of 1 jumps at every step, so consecutive draws are independent."""
    path = stationary_bootstrap_indices(n_obs=800, block_length=1.0, n_paths=1, seed=SEED)[0]
    steps = np.diff(path) % 800
    # Under iid sampling, P(next == current + 1) is 1/800; nothing like the >0.95 above.
    assert np.mean(steps == 1) < 0.05


def test_paths_are_reproducible_and_distinct() -> None:
    first = stationary_bootstrap_indices(400, 20.0, 5, seed=SEED)
    second = stationary_bootstrap_indices(400, 20.0, 5, seed=SEED)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1]), "paths within one draw must differ"


def test_a_different_seed_gives_different_paths() -> None:
    assert not np.array_equal(
        stationary_bootstrap_indices(400, 20.0, 3, seed=SEED),
        stationary_bootstrap_indices(400, 20.0, 3, seed=SEED + 1),
    )


@pytest.mark.parametrize(
    ("n_obs", "block_length", "n_paths"),
    [(1, 10.0, 1), (100, 0.5, 1), (100, np.inf, 1), (100, 10.0, 0)],
)
def test_rejects_degenerate_arguments(n_obs: int, block_length: float, n_paths: int) -> None:
    with pytest.raises(ResamplingError):
        stationary_bootstrap_indices(n_obs, block_length, n_paths, seed=SEED)


# --- block-length selection -----------------------------------------------------


def test_block_length_rises_with_dependence() -> None:
    """The whole point of an automatic rule: more persistence must give longer blocks."""
    weak = optimal_block_length(_ar1(2000, phi=0.05))
    strong = optimal_block_length(_ar1(2000, phi=0.75))
    assert strong.block_length > weak.block_length, (weak.block_length, strong.block_length)


def test_block_length_is_bounded_and_reported_with_its_workings() -> None:
    estimate = optimal_block_length(_ar1(1500, phi=0.9))
    assert isinstance(estimate, BlockLengthEstimate)
    assert 1.0 <= estimate.block_length <= 3.0 * 1500 ** (1.0 / 3.0)
    assert estimate.n_obs == 1500
    assert estimate.bandwidth >= 1


def test_block_length_rejects_a_series_too_short_to_estimate() -> None:
    with pytest.raises(ResamplingError, match="at least 16"):
        optimal_block_length(np.arange(10.0))


def test_block_length_rejects_a_constant_series() -> None:
    with pytest.raises(ResamplingError, match="zero variance"):
        optimal_block_length(np.ones(200))


# --- conditioning ---------------------------------------------------------------


def test_conditional_jumps_never_change_label() -> None:
    """Within a path, every jump lands on a position carrying the current label."""
    labels = np.repeat([0, 1, 2, 3], 200)
    paths = conditional_bootstrap_indices(labels, block_length=10.0, n_paths=4, seed=SEED)

    for path in paths:
        for step in range(1, path.shape[0]):
            previous, current = path[step - 1], path[step]
            advanced = current == (previous + 1) % labels.shape[0]
            if not advanced:
                assert labels[current] == labels[previous], (
                    f"jump at step {step} crossed from label {labels[previous]} "
                    f"to {labels[current]}"
                )


def test_conditional_paths_are_valid_indices_and_reproducible() -> None:
    labels = np.repeat([0, 1], 150)
    first = conditional_bootstrap_indices(labels, 12.0, 3, seed=SEED)
    second = conditional_bootstrap_indices(labels, 12.0, 3, seed=SEED)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0 and first.max() < labels.shape[0]


def test_conditioning_refuses_a_label_with_too_few_observations() -> None:
    """A label seen once would be 'resampled' by returning the same day forever."""
    labels = np.array([0] * 200 + [1])
    with pytest.raises(ResamplingError, match="fewer than 2 observations"):
        conditional_bootstrap_indices(labels, 10.0, 2, seed=SEED)


# --- moment comparison ----------------------------------------------------------


def test_moment_report_covers_the_battery_and_finds_the_real_series() -> None:
    """Sanity: resampling a series and comparing it to itself should mostly agree."""
    returns = _garch_like(1200)
    paths = stationary_bootstrap_indices(returns.shape[0], 20.0, 200, seed=SEED)

    report = compare_moments(returns, paths)
    names = {c.name for c in report.comparisons}
    assert {"std", "skewness", "excess_kurtosis", "abs_autocorr_lag1", "max_drawdown"} <= names
    # Mean and standard deviation are preserved by any resampling of the same values.
    for name in ("mean", "std"):
        comparison = next(c for c in report.comparisons if c.name == name)
        assert comparison.real_inside_interval, name


def test_short_blocks_destroy_volatility_clustering() -> None:
    """A negative control: with block length 1 the bootstrap is iid and clustering must vanish.

    This is what makes the moment report meaningful. If clustering survived even iid resampling,
    the statistic would not be measuring what it claims to.
    """
    returns = _garch_like(1500)
    iid = stationary_bootstrap_indices(returns.shape[0], 1.0, 200, seed=SEED)

    report = compare_moments(returns, iid)
    clustering = next(c for c in report.comparisons if c.name == "abs_autocorr_lag1")
    assert not clustering.real_inside_interval
    assert clustering.real > clustering.synthetic_p97_5
    assert clustering.real_percentile > 99.0


def test_moment_comparison_rejects_mismatched_lengths() -> None:
    with pytest.raises(MomentError, match="must match"):
        compare_moments(np.zeros(100), np.zeros((5, 50), dtype=np.int64))


def test_max_drawdown_is_a_positive_fraction() -> None:
    returns = np.array([0.1, -0.5, 0.2, -0.1])
    value = max_drawdown(returns)
    assert 0.0 < value < 1.0
    # Peak after the first step is 1.1; trough after the second is 0.55; drawdown = 0.5.
    assert value == pytest.approx(0.5, rel=1e-9)
