"""The diagnostics decide whether a null result is real, so each is checked against a known answer.

If ``permutation_test`` returned small p-values on noise, or ``learning_curve`` sloped upward on
data with no signal, this project would conclude that fragility is predictable when it is not. Every
test below therefore constructs a case whose correct answer is known before the function runs.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.stress.diagnosis import (
    feature_collinearity,
    influence,
    learning_curve,
    permutation_test,
    target_shape,
    trimmed_scores,
)

N_ROWS = 120


def test_a_symmetric_target_has_no_skew_and_a_heavy_one_does() -> None:
    rng = np.random.default_rng(42)
    symmetric = target_shape(np.asarray(rng.normal(size=5000)))
    assert abs(symmetric["skewness"]) < 0.2
    assert abs(symmetric["excess_kurtosis"]) < 0.5

    heavy = np.concatenate([rng.normal(size=999), np.array([500.0])])
    result = target_shape(heavy)
    assert result["skewness"] > 5.0
    # One row in a thousand owning most of the sum of squares is the situation that makes an R^2
    # a statement about that row rather than about the population.
    assert result["variance_share_top_1"] > 0.9


def test_variance_shares_are_cumulative_and_bounded() -> None:
    rng = np.random.default_rng(1)
    result = target_shape(np.asarray(rng.normal(size=500)))
    assert 0.0 <= result["variance_share_top_1"] <= result["variance_share_top_5"]
    assert result["variance_share_top_5"] <= result["variance_share_top_10"] <= 1.0


def test_a_duplicated_column_is_reported_as_perfectly_correlated() -> None:
    """The defect this function was written to catch: the same column twice under two names."""
    rng = np.random.default_rng(2)
    base = np.asarray(rng.normal(size=(N_ROWS, 3)))
    features = np.column_stack([base, base[:, 0]])
    result = feature_collinearity(features, ["a", "b", "c", "a_again"])
    assert result["max_abs_pairwise_correlation"] == pytest.approx(1.0)
    assert set(cast(list[str], result["worst_pair"])) == {"a", "a_again"}
    assert cast(float, result["condition_number"]) > 1e6


def test_independent_columns_are_well_conditioned() -> None:
    rng = np.random.default_rng(3)
    result = feature_collinearity(
        np.asarray(rng.normal(size=(N_ROWS, 5))), ["a", "b", "c", "d", "e"]
    )
    assert cast(float, result["condition_number"]) < 5.0


def test_a_constant_column_is_dropped_rather_than_dividing_by_zero() -> None:
    rng = np.random.default_rng(4)
    features = np.column_stack([rng.normal(size=(N_ROWS, 2)), np.ones(N_ROWS)])
    result = feature_collinearity(features, ["a", "b", "constant"])
    assert result["n_constant_features_dropped"] == 1
    assert np.isfinite(cast(float, result["condition_number"]))


def test_permutation_test_finds_no_significance_in_noise() -> None:
    """The single most important test here. Noise must not produce a small p-value.

    If it did, every null result this phase reports could be an artefact of the test rather than a
    property of the data.
    """
    rng = np.random.default_rng(5)
    features = np.asarray(rng.normal(size=(N_ROWS, 5)))
    target = np.asarray(rng.normal(size=N_ROWS))
    result = permutation_test(
        features, target, [f"s{i}" for i in range(N_ROWS)],
        kind="ridge", seed=42, repeats=60,
    )
    assert result["p_value_spearman"] > 0.05
    assert result["p_value_r2"] > 0.05


def test_permutation_test_finds_significance_in_a_real_relationship() -> None:
    """The opposite control: a genuine linear relationship must clear the null comfortably."""
    rng = np.random.default_rng(6)
    features = np.asarray(rng.normal(size=(N_ROWS, 5)))
    target = 3.0 * features[:, 0] + np.asarray(rng.normal(scale=0.5, size=N_ROWS))
    result = permutation_test(
        features, target, [f"s{i}" for i in range(N_ROWS)],
        kind="ridge", seed=42, repeats=60,
    )
    assert result["p_value_spearman"] < 0.05
    assert result["kind_spearman"] > result["null_abs_spearman_p95"]


def test_the_learning_curve_climbs_where_more_data_helps() -> None:
    rng = np.random.default_rng(7)
    features = np.asarray(rng.normal(size=(N_ROWS, 5)))
    target = 2.0 * features[:, 0] + np.asarray(rng.normal(scale=2.0, size=N_ROWS))
    curve = learning_curve(
        features, target, [f"s{i}" for i in range(N_ROWS)],
        sizes=(30, 60, 110), kind="ridge", seed=42, repeats=5,
    )
    assert [point["n"] for point in curve] == [30.0, 60.0, 110.0]
    assert curve[-1]["spearman_mean"] > curve[0]["spearman_mean"]


def test_the_learning_curve_stays_flat_on_noise() -> None:
    """More rows of nothing is still nothing. A rising curve here would be the diagnostic lying."""
    rng = np.random.default_rng(8)
    features = np.asarray(rng.normal(size=(N_ROWS, 5)))
    target = np.asarray(rng.normal(size=N_ROWS))
    curve = learning_curve(
        features, target, [f"s{i}" for i in range(N_ROWS)],
        sizes=(30, 110), kind="ridge", seed=42, repeats=5,
    )
    assert abs(curve[-1]["spearman_mean"] - curve[0]["spearman_mean"]) < 0.25


def test_a_single_extreme_row_is_identified_as_the_influential_one() -> None:
    """One row far outside the rest must top the influence ranking, by name."""
    rng = np.random.default_rng(9)
    features = np.asarray(rng.normal(size=(40, 3)))
    target = np.asarray(rng.normal(scale=0.1, size=40))
    target[7] = 50.0
    names = [f"s{i}" for i in range(40)]
    result = influence(features, target, names, kind="ridge", seed=42, top=3)
    ranked = [row["name"] for row in cast(list[dict[str, object]], result["most_influential"])]
    assert ranked[0] == "s7"


def test_trimming_reports_its_own_sample_size_at_every_level() -> None:
    rng = np.random.default_rng(10)
    target = np.asarray(rng.normal(size=50))
    predictions = target + rng.normal(scale=0.1, size=50)
    result = trimmed_scores(target, predictions, (0, 5, 10))
    assert result["0"]["n"] == 50.0
    assert result["5"]["n"] == 45.0
    assert result["10"]["n"] == 40.0
