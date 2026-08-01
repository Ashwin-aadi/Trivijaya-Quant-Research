"""Tests for expanding-window regime labelling.

The load-bearing test in this file is ``test_filtered_label_ignores_the_future``. Everything else
checks that the machinery is correct; that one checks that it is *causal*, which is the entire
reason Phase 2.0 was designed the way it was. If it ever fails, regime labels are leaking and no
downstream fragility number means anything.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.stress.regimes import (
    FEATURE_NAMES,
    MIN_TRAIN_SESSIONS,
    RegimeError,
    bayesian_information_criterion,
    build_features,
    canonical_permutation,
    expanding_window_labels,
    feature_matrix,
    fit_regime_model,
    forward_filter,
    n_free_parameters,
)

SEED = 42


def _synthetic_closes(n: int, seed: int = SEED, switch_every: int = 90) -> pl.DataFrame:
    """A price series alternating between calm and violent regimes every ``switch_every`` sessions.

    Deliberately synthetic. These tests check the mechanics of causality and canonicalisation,
    which must hold for any input; they are not evidence about Indian markets.

    It alternates repeatedly rather than switching once, because a single well-separated switch is
    too easy: every decoder agrees on it, and the negative control below would then pass vacuously.
    Frequent switching creates genuinely ambiguous sessions near the boundaries, which is where a
    forward-only decoder and a whole-sequence decoder actually part company.
    """
    rng = np.random.default_rng(seed)
    returns = np.empty(n, dtype=np.float64)
    for start_index in range(0, n, switch_every):
        stop = min(start_index + switch_every, n)
        calm = (start_index // switch_every) % 2 == 0
        loc, scale = (0.0005, 0.004) if calm else (-0.001, 0.018)
        returns[start_index:stop] = rng.normal(loc, scale, stop - start_index)
    closes = 1000.0 * np.exp(np.cumsum(returns))
    start = date(2015, 1, 1)
    return pl.DataFrame(
        {
            "session_date": [start + timedelta(days=i) for i in range(n)],
            "close": closes.tolist(),
        }
    )


# --- features -------------------------------------------------------------------


def test_features_use_only_past_sessions() -> None:
    """Changing a future close must not change any earlier feature row.

    This is the point-in-time guarantee at the feature layer. A single negative shift anywhere in
    build_features would break it.
    """
    frame = _synthetic_closes(300)
    base = build_features(frame)

    perturbed = frame.with_columns(
        pl.when(pl.col("session_date") >= frame["session_date"][250])
        .then(pl.col("close") * 3.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    after = build_features(perturbed)

    cutoff = frame["session_date"][250]
    left = base.filter(pl.col("session_date") < cutoff)
    right = after.filter(pl.col("session_date") < cutoff)
    assert left.height > 0
    for name in FEATURE_NAMES:
        np.testing.assert_allclose(left[name].to_numpy(), right[name].to_numpy(), rtol=1e-12)


def test_features_drop_the_warmup_rather_than_imputing_it() -> None:
    frame = _synthetic_closes(60)
    built = build_features(frame, window=21)
    # 60 closes -> 59 returns -> 39 rows with a full 21-session trailing window.
    assert built.height == 60 - 21
    assert built["session_date"][0] == frame["session_date"][21]


def test_features_reject_an_impossible_window() -> None:
    with pytest.raises(RegimeError):
        build_features(_synthetic_closes(60), window=1)


def test_features_reject_too_short_a_series() -> None:
    with pytest.raises(RegimeError):
        build_features(_synthetic_closes(10), window=21)


# --- the filtered decode: the causality guarantee --------------------------------


def test_filtered_label_ignores_the_future() -> None:
    """The label at t must be identical whether or not sessions after t exist in the input.

    This is what separates filtering from smoothing. Run the same fitted model over a truncated
    sequence and over the full one; every overlapping label must match exactly. hmmlearn's
    `predict` (Viterbi) and `predict_proba` (smoothed) both fail this test by construction, which
    is why neither is used.
    """
    frame = _synthetic_closes(1400)
    features = build_features(frame)
    matrix = feature_matrix(features)
    model = fit_regime_model(matrix[:600], k=2, fitted_through=date(2016, 1, 1), seed=SEED)

    full = model.filtered_states(matrix)
    for cutoff in (700, 900, 1100):
        truncated = model.filtered_states(matrix[:cutoff])
        np.testing.assert_array_equal(truncated, full[:cutoff])


def test_the_forbidden_decoder_produces_different_labels() -> None:
    """Negative control: the decoder choice is load-bearing, not cosmetic.

    Without this, ``test_filtered_label_ignores_the_future`` could pass vacuously — if filtered and
    smoothed labels coincided everywhere, our decoder choice would prove nothing.

    What is asserted is the honest claim and no more: on the *same* fitted model and the *same*
    sequence, the smoothed decoder assigns different states from the filtered one. Smoothing
    conditions on ``obs_1..T``, so those differing sessions are labelled using information that did
    not exist at the session being labelled.

    Note deliberately not asserted: that truncating the sequence moves the smoothed posteriors by
    much. Measured on this fixture, it does not — the largest change is ~7e-7 and confined to the
    final row, because these regimes are persistent and well separated, so the backward pass adds
    almost nothing more than a few observations from the end. The leakage is structural rather than
    numerically dramatic, and overstating it here would be the kind of convenient claim this
    repository exists to catch.
    """
    from hmmlearn.hmm import GaussianHMM

    matrix = feature_matrix(build_features(_synthetic_closes(1400)))
    raw = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=200, random_state=SEED, tol=1e-4
    )
    raw.fit(matrix[:600])
    ours = fit_regime_model(matrix[:600], k=2, fitted_through=date(2016, 1, 1), seed=SEED)

    smoothed = np.asarray(raw.predict_proba(matrix)).argmax(axis=1)
    filtered = ours.filtered_states(matrix)
    disagreements = int(np.sum(smoothed != filtered))
    assert disagreements > 0, (
        "filtered and smoothed labels agreed everywhere on this fixture, so the causality test "
        "above is not discriminating and the fixture needs to be harder"
    )


def test_forward_filter_rows_are_normalised_posteriors() -> None:
    log_start = np.log(np.array([0.5, 0.5]))
    log_trans = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    rng = np.random.default_rng(SEED)
    frame_ll = rng.normal(-2.0, 1.0, size=(50, 2))

    out, evidence = forward_filter(log_start, log_trans, frame_ll)
    np.testing.assert_allclose(np.sum(np.exp(out), axis=1), np.ones(50), rtol=1e-10)
    assert np.isfinite(evidence)


def test_forward_filter_first_row_is_prior_times_emission() -> None:
    log_start = np.log(np.array([0.3, 0.7]))
    log_trans = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))
    frame_ll = np.array([[-1.0, -2.0], [-1.0, -1.0]])

    out, _ = forward_filter(log_start, log_trans, frame_ll)
    unnormalised = log_start + frame_ll[0]
    expected = unnormalised - np.log(np.sum(np.exp(unnormalised)))
    np.testing.assert_allclose(out[0], expected, rtol=1e-12)


# --- canonicalisation -----------------------------------------------------------


def test_canonical_permutation_orders_by_volatility_ascending() -> None:
    means = np.array([[0.5, 0.0], [-1.2, 0.0], [0.1, 0.0]])
    np.testing.assert_array_equal(canonical_permutation(means), np.array([1, 2, 0]))


def test_fitted_model_comes_back_in_canonical_order() -> None:
    """State 0 is always the calmest, so labels are comparable across refits."""
    matrix = feature_matrix(build_features(_synthetic_closes(1400)))
    model = fit_regime_model(matrix[:800], k=3, fitted_through=date(2017, 1, 1), seed=SEED)
    volatility_means = model.means[:, 0]
    assert np.all(np.diff(volatility_means) > 0), volatility_means
    np.testing.assert_allclose(model.startprob.sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(model.transmat.sum(axis=1), np.ones(3), rtol=1e-10)


def test_forward_recursion_matches_hmmlearn_evidence() -> None:
    """Cross-check the hand-written forward pass against hmmlearn on identical parameters.

    The decode is implemented here rather than taken from the library, so it needs independent
    corroboration: a filter that is causal but arithmetically wrong would pass every other test in
    this file. Both compute log P(obs_1..T), which the forward normalisers accumulate exactly.

    This also establishes that canonicalisation is a pure relabelling — the permuted parameters
    reproduce the unpermuted model's evidence to floating-point tolerance.
    """
    from hmmlearn.hmm import GaussianHMM

    matrix = feature_matrix(build_features(_synthetic_closes(1400)))[:800]
    ours = fit_regime_model(matrix, k=3, fitted_through=date(2017, 1, 1), seed=SEED)

    # Load OUR canonicalised parameters into hmmlearn and score with its own machinery. Fitting a
    # second model independently would compare two different local optima, not two implementations.
    mirror = GaussianHMM(n_components=3, covariance_type="full")
    mirror.startprob_ = ours.startprob
    mirror.transmat_ = ours.transmat
    mirror.means_ = ours.means
    mirror.covars_ = ours.covars

    reference = float(mirror.score(matrix))
    assert ours.sequence_log_evidence(matrix) == pytest.approx(reference, rel=1e-9)
    # And the permuted parameters reproduce the unpermuted fit's likelihood, so canonicalisation is
    # a relabelling and not a different model.
    assert ours.log_likelihood == pytest.approx(reference, rel=1e-9)


# --- the training floor ---------------------------------------------------------


def test_fit_refuses_to_run_below_the_floor() -> None:
    matrix = feature_matrix(build_features(_synthetic_closes(600)))
    with pytest.raises(RegimeError, match="floor"):
        fit_regime_model(
            matrix[: MIN_TRAIN_SESSIONS - 1], k=2, fitted_through=date(2016, 1, 1), seed=SEED
        )


def test_sessions_before_the_first_eligible_refit_are_absent_not_null() -> None:
    """An unlabelled session must not appear with a null, which a join could treat as a label."""
    frame = _synthetic_closes(1400)
    features = build_features(frame)
    dates = features["session_date"].to_list()
    refits = [dates[300], dates[700], dates[1000]]

    labels, models = expanding_window_labels(features, refits, k=2, seed=SEED, min_train=600)

    assert labels["state"].null_count() == 0
    # The 300-row refit is below the 600 floor, so labelling starts at the second boundary.
    assert labels["session_date"].min() == dates[700]
    assert len(models) == 2


def test_every_label_comes_from_a_model_fit_strictly_before_it() -> None:
    """The parameter half of the causal guarantee, asserted rather than assumed."""
    features = build_features(_synthetic_closes(1600))
    dates = features["session_date"].to_list()
    refits = [dates[700], dates[1000], dates[1300]]

    labels, models = expanding_window_labels(features, refits, k=2, seed=SEED, min_train=600)

    by_refit = {m.fitted_through: m for m in models}
    for row in labels.iter_rows(named=True):
        model = next(m for m in models if m.n_train == row["n_train"])
        assert model.fitted_through < row["session_date"]
    assert len(by_refit) == len(models)


def test_expanding_window_raises_when_nothing_clears_the_floor() -> None:
    features = build_features(_synthetic_closes(900))
    dates = features["session_date"].to_list()
    with pytest.raises(RegimeError, match="nothing was labelled"):
        expanding_window_labels(features, [dates[100]], k=2, seed=SEED, min_train=600)


# --- BIC ------------------------------------------------------------------------


def test_free_parameter_count_matches_the_written_formula() -> None:
    # K=3, D=2: startprob 2 + transmat 6 + means 6 + covars 9 = 23.
    assert n_free_parameters(3, 2) == 23
    # K=2, D=2: 1 + 2 + 4 + 6 = 13.
    assert n_free_parameters(2, 2) == 13


def test_bic_penalises_the_larger_model_on_identical_data() -> None:
    """Same data, more states: the penalty term must be strictly larger."""
    matrix = feature_matrix(build_features(_synthetic_closes(1400)))[:900]
    small = fit_regime_model(matrix, k=2, fitted_through=date(2017, 6, 1), seed=SEED)
    large = fit_regime_model(matrix, k=4, fitted_through=date(2017, 6, 1), seed=SEED)

    penalty_small = n_free_parameters(2, 2) * np.log(900)
    penalty_large = n_free_parameters(4, 2) * np.log(900)
    assert penalty_large > penalty_small
    assert bayesian_information_criterion(small, matrix) == pytest.approx(
        -2.0 * small.log_likelihood + penalty_small, rel=1e-9
    )
    assert bayesian_information_criterion(large, matrix) == pytest.approx(
        -2.0 * large.log_likelihood + penalty_large, rel=1e-9
    )


def test_restarts_never_lower_the_training_likelihood() -> None:
    """Restarts take the best of several EM runs, so more of them cannot fit worse.

    This guards a real failure observed on the burn-in window: at K=3 a single start converged to a
    degenerate solution with two duplicate states and a log-likelihood *below* the K=2 fit, which
    is impossible for a correctly converged nested family and corrupted the BIC comparison.
    """
    matrix = feature_matrix(build_features(_synthetic_closes(1600)))[:900]
    single = fit_regime_model(
        matrix, k=3, fitted_through=date(2017, 6, 1), seed=SEED, n_restarts=1
    )
    many = fit_regime_model(matrix, k=3, fitted_through=date(2017, 6, 1), seed=SEED, n_restarts=8)
    assert many.log_likelihood >= single.log_likelihood - 1e-9


def test_restart_sweep_is_deterministic() -> None:
    """Restart seeds are derived from the global seed, so the sweep is reproducible."""
    matrix = feature_matrix(build_features(_synthetic_closes(1600)))[:900]
    first = fit_regime_model(matrix, k=3, fitted_through=date(2017, 6, 1), seed=SEED)
    second = fit_regime_model(matrix, k=3, fitted_through=date(2017, 6, 1), seed=SEED)
    assert first.log_likelihood == pytest.approx(second.log_likelihood, rel=1e-12)
    np.testing.assert_allclose(first.means, second.means, rtol=1e-12)


def test_fit_rejects_a_nonsensical_restart_count() -> None:
    matrix = feature_matrix(build_features(_synthetic_closes(1600)))[:900]
    with pytest.raises(RegimeError, match="n_restarts"):
        fit_regime_model(matrix, k=2, fitted_through=date(2017, 6, 1), seed=SEED, n_restarts=0)


def test_labels_are_reproducible_under_the_same_seed() -> None:
    features = build_features(_synthetic_closes(1600))
    dates = features["session_date"].to_list()
    refits = [dates[700], dates[1100]]

    first, _ = expanding_window_labels(features, refits, k=3, seed=SEED, min_train=600)
    second, _ = expanding_window_labels(features, refits, k=3, seed=SEED, min_train=600)
    assert first.equals(second)
