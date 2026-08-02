"""The predictor's scoring must be able to report failure. These tests check that it does.

The risk in this module is not that the model is weak — a weak model is a legitimate result. The
risk is that the *scoring* flatters it: a baseline that has seen the test fold, an R^2 measured
against the test mean, or an importance that rewards noise. Each is pinned below with a case whose
answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stress.predictor import (
    cross_validate,
    permutation_importance,
    spearman,
)

N_ROWS = 200


def test_spearman_of_a_series_with_itself_is_one() -> None:
    rng = np.random.default_rng(42)
    values = np.asarray(rng.normal(size=N_ROWS))
    assert spearman(values, values) == pytest.approx(1.0)


def test_spearman_of_a_reversed_series_is_minus_one() -> None:
    rng = np.random.default_rng(42)
    values = np.asarray(rng.normal(size=N_ROWS))
    assert spearman(values, -values) == pytest.approx(-1.0)


def test_a_target_with_no_signal_scores_no_better_than_the_baseline() -> None:
    """Pure noise against pure noise. R^2 must come out at or below zero.

    This is the test that matters most. If a model fed unrelated features returned a positive
    out-of-fold R^2, every headline number this module produces would be uninterpretable.
    """
    rng = np.random.default_rng(0)
    features = np.asarray(rng.normal(size=(N_ROWS, 6)))
    target = np.asarray(rng.normal(size=N_ROWS))
    result, _ = cross_validate(
        features, target, [f"s{i}" for i in range(N_ROWS)], target_name="noise", seed=42
    )
    assert result.r2_model <= 0.05        # small positive slack for fold-level luck
    assert result.mae_model >= result.mae_baseline * 0.95


def test_a_learnable_target_beats_the_baseline_clearly() -> None:
    """The opposite control: a target that is a smooth function of one feature must be learnable.

    Without this, the noise test above would also pass on a model that always predicts the mean.
    """
    rng = np.random.default_rng(1)
    features = np.asarray(rng.normal(size=(N_ROWS, 6)))
    target = 3.0 * features[:, 0] + 0.1 * np.asarray(rng.normal(size=N_ROWS))
    result, _ = cross_validate(
        features, target, [f"s{i}" for i in range(N_ROWS)], target_name="linear", seed=42
    )
    assert result.r2_model > 0.8
    assert result.spearman > 0.9
    assert result.mae_model < result.mae_baseline


def test_rows_with_a_missing_target_are_dropped_rather_than_imputed() -> None:
    """A strategy whose fragility could not be computed must not be given a fabricated one."""
    rng = np.random.default_rng(2)
    features = np.asarray(rng.normal(size=(N_ROWS, 4)))
    target = np.asarray(rng.normal(size=N_ROWS))
    target[:20] = np.nan
    result, predictions = cross_validate(
        features, target, [f"s{i}" for i in range(N_ROWS)], target_name="gappy", seed=42
    )
    assert result.n_rows == N_ROWS - 20
    assert predictions.shape[0] == N_ROWS - 20


def test_the_only_informative_feature_carries_the_importance() -> None:
    """Shuffling the feature the target is built from must cost more than shuffling the others."""
    rng = np.random.default_rng(3)
    features = np.asarray(rng.normal(size=(N_ROWS, 4)))
    target = 3.0 * features[:, 2] + 0.1 * np.asarray(rng.normal(size=N_ROWS))
    columns = ["a", "b", "signal", "d"]
    importance = permutation_importance(features, target, columns, seed=42, repeats=3)
    assert importance["signal"] > max(importance[c] for c in ("a", "b", "d"))


def test_the_same_seed_gives_the_same_score_twice() -> None:
    """Reproducibility is a hard requirement, and k-fold shuffling is a stochastic operation."""
    rng = np.random.default_rng(4)
    features = np.asarray(rng.normal(size=(N_ROWS, 5)))
    target = np.asarray(features[:, 0] + rng.normal(scale=0.5, size=N_ROWS))
    names = [f"s{i}" for i in range(N_ROWS)]
    first, _ = cross_validate(features, target, names, target_name="repeat", seed=42)
    second, _ = cross_validate(features, target, names, target_name="repeat", seed=42)
    assert first.r2_model == second.r2_model
    assert first.spearman == second.spearman
