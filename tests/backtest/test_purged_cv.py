"""Tests for purged k-fold cross-validation with an embargo.

This is the most load-bearing test file in the backtest package. Purging and embargoing are the
only thing standing between an overlapping-label time series and a model that has been quietly
told the answer, and both are silent when they fail: a leaky split produces a *better* score, not
an error. So the tests come in two halves.

The first half checks that ``purged_kfold_indices`` produces splits ``assert_no_leakage`` accepts,
across a spread of parameter combinations rather than one convenient one.

The second half is the half that matters. A checker that never fires is worth nothing, and a
checker only tested against splits that already satisfy it has never been shown to fire at all.
So we hand-build folds that deliberately break the purge and the embargo and assert that
``assert_no_leakage`` raises on each. If any of those tests ever start passing silently, the
checker has stopped checking and every cross-validated number downstream is unverified.
"""

import numpy as np
import pytest

from src.backtest.purged_cv import Fold, assert_no_leakage, purged_kfold_indices

# (n_samples, n_splits, label_horizon, embargo). Deliberately varied: a long horizon with no
# embargo, a zero horizon, a fold count equal to the sample count, and the project defaults.
SPLIT_CASES = [
    (100, 5, 1, 5),
    (100, 3, 10, 0),
    (250, 5, 5, 5),
    (37, 4, 3, 2),
    (10, 10, 1, 1),
    (1000, 8, 20, 10),
    (60, 2, 0, 0),
]


# --- the splits themselves ------------------------------------------------------


@pytest.mark.parametrize(("n_samples", "n_splits", "label_horizon", "embargo"), SPLIT_CASES)
def test_test_folds_are_contiguous_and_cover_every_index(
    n_samples: int,
    n_splits: int,
    label_horizon: int,
    embargo: int,
) -> None:
    """Every observation is tested exactly once, and each test fold is an unbroken time block.

    Contiguity is not cosmetic. Purging is defined relative to a single [start, end] interval, so
    a fold made of scattered indices would leave gaps the purge never covers.
    """
    folds = purged_kfold_indices(n_samples, n_splits, label_horizon, embargo)
    assert len(folds) == n_splits

    for fold in folds:
        block = fold.test_indices
        assert block.size > 0
        # Consecutive integers from first to last: no holes, no reordering.
        assert np.array_equal(block, np.arange(int(block[0]), int(block[-1]) + 1))

    covered = np.sort(np.concatenate([fold.test_indices for fold in folds]))
    assert np.array_equal(covered, np.arange(n_samples))


@pytest.mark.parametrize(("n_samples", "n_splits", "label_horizon", "embargo"), SPLIT_CASES)
def test_generated_folds_survive_the_independent_leakage_check(
    n_samples: int,
    n_splits: int,
    label_horizon: int,
    embargo: int,
) -> None:
    """The splitter's output satisfies the property, checked by code that does not share its logic.

    ``assert_no_leakage`` is written against the invariant rather than against how the splitter
    computes it, so this is a real cross-check and not a restatement.
    """
    folds = purged_kfold_indices(n_samples, n_splits, label_horizon, embargo)
    for fold in folds:
        assert_no_leakage(fold, label_horizon=label_horizon, embargo=embargo)


@pytest.mark.parametrize(("n_samples", "n_splits", "label_horizon", "embargo"), SPLIT_CASES)
def test_no_training_index_sits_inside_the_purged_or_embargoed_window(
    n_samples: int,
    n_splits: int,
    label_horizon: int,
    embargo: int,
) -> None:
    """Stated as a plain interval test, independent of both the splitter and the checker."""
    folds = purged_kfold_indices(n_samples, n_splits, label_horizon, embargo)
    for fold in folds:
        test_start = int(fold.test_indices[0])
        test_end = int(fold.test_indices[-1])
        forbidden_low = test_start - label_horizon
        forbidden_high = test_end + embargo
        for index in fold.train_indices:
            assert not (forbidden_low <= int(index) <= forbidden_high)


# --- the checker must actually fire ---------------------------------------------


def test_assert_no_leakage_raises_on_an_embargo_violation() -> None:
    """A training point just past the fold end is inside the embargo shadow and must be rejected.

    This is the negative control for the embargo. Index 21 sits two positions after a fold ending
    at 19; with a five-period embargo it is illegal, and with no embargo requested it is fine.
    Both halves are asserted so the test cannot pass by the checker raising unconditionally.
    """
    fold = Fold(train_indices=np.array([21]), test_indices=np.arange(10, 20))

    with pytest.raises(ValueError, match="embargo"):
        assert_no_leakage(fold, label_horizon=1, embargo=5)

    # Same fold, no embargo asked for: index 21 is legitimate training data.
    assert_no_leakage(fold, label_horizon=1, embargo=0)


def test_assert_no_leakage_raises_when_a_training_label_reaches_into_the_test_fold() -> None:
    """The negative control for purging.

    A label formed at index 8 over a five-period horizon resolves at index 13, which is inside a
    test fold spanning [10, 19]. Training on it means training on the test period's outcome. With
    a one-period horizon the same label resolves at 9, safely before the fold, and is allowed.
    """
    fold = Fold(train_indices=np.array([8]), test_indices=np.arange(10, 20))

    with pytest.raises(ValueError, match="label window"):
        assert_no_leakage(fold, label_horizon=5, embargo=5)

    # Shorter horizon: the label resolves before the fold opens, so nothing leaks.
    assert_no_leakage(fold, label_horizon=1, embargo=5)


def test_mutating_a_valid_fold_makes_the_checker_fire() -> None:
    """Take a fold the checker accepts, move one index into the shadow, and it must reject it.

    Written this way because it is the closest thing to a mutation test we can run cheaply: it
    proves the checker's verdict depends on the data rather than on the fold merely existing.
    """
    folds = purged_kfold_indices(n_samples=100, n_splits=5, label_horizon=1, embargo=5)
    clean = folds[0]
    assert_no_leakage(clean, label_horizon=1, embargo=5)

    test_end = int(clean.test_indices[-1])
    smuggled = np.append(clean.train_indices, test_end + 1)
    tampered = Fold(train_indices=smuggled, test_indices=clean.test_indices)

    with pytest.raises(ValueError, match="embargo"):
        assert_no_leakage(tampered, label_horizon=1, embargo=5)


def test_assert_no_leakage_rejects_an_empty_test_fold() -> None:
    """An empty test fold has no [start, end] to purge around, so it is an error, not a no-op."""
    fold = Fold(train_indices=np.arange(0, 10), test_indices=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="empty test fold"):
        assert_no_leakage(fold, label_horizon=1, embargo=5)


# --- Fold construction ----------------------------------------------------------


def test_fold_rejects_overlapping_train_and_test() -> None:
    """The crudest leak of all — the same observation in both sets — is refused at construction."""
    with pytest.raises(ValueError, match="overlap"):
        Fold(train_indices=np.arange(0, 15), test_indices=np.arange(10, 20))


def test_fold_accepts_a_disjoint_split() -> None:
    """The complement of the test above, so the constructor is not simply rejecting everything."""
    fold = Fold(train_indices=np.arange(0, 5), test_indices=np.arange(10, 20))
    assert fold.train_indices.size == 5
    assert fold.test_indices.size == 10


# --- argument validation --------------------------------------------------------


def test_single_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two folds"):
        purged_kfold_indices(n_samples=100, n_splits=1)


def test_more_folds_than_samples_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot split"):
        purged_kfold_indices(n_samples=3, n_splits=5)


@pytest.mark.parametrize(("label_horizon", "embargo"), [(-1, 5), (1, -1), (-2, -3)])
def test_negative_horizon_or_embargo_is_rejected(label_horizon: int, embargo: int) -> None:
    """A negative horizon would purge backwards; a negative embargo would *add* adjacent data."""
    with pytest.raises(ValueError, match="non-negative"):
        purged_kfold_indices(
            n_samples=100,
            n_splits=5,
            label_horizon=label_horizon,
            embargo=embargo,
        )


# --- monotonicity ---------------------------------------------------------------


@pytest.mark.parametrize(("n_samples", "n_splits", "label_horizon"), [(200, 5, 1), (120, 4, 10)])
def test_a_longer_embargo_never_yields_more_training_data(
    n_samples: int,
    n_splits: int,
    label_horizon: int,
) -> None:
    """Each fold's training set must shrink, or stay put, as the embargo lengthens.

    This is the direction-of-effect check. An off-by-one sign in the mask would still produce
    plausible-looking folds that pass every other test here while quietly handing the model more
    data the stricter you asked it to be.
    """
    previous: list[set[int]] | None = None
    for embargo in (0, 1, 5, 20):
        folds = purged_kfold_indices(n_samples, n_splits, label_horizon, embargo)
        current = [{int(i) for i in fold.train_indices} for fold in folds]
        if previous is not None:
            for stricter, looser in zip(current, previous, strict=True):
                assert stricter <= looser
        previous = current


def test_a_longer_embargo_strictly_shrinks_the_total_training_set() -> None:
    """Subset alone would be satisfied by an embargo that does nothing at all.

    The last fold ends at the final observation, so its embargo shadow falls off the end and
    removes nothing. Every earlier fold must lose points, so the total has to fall.
    """
    without = purged_kfold_indices(n_samples=200, n_splits=5, label_horizon=1, embargo=0)
    with_embargo = purged_kfold_indices(n_samples=200, n_splits=5, label_horizon=1, embargo=20)

    total_without = sum(fold.train_indices.size for fold in without)
    total_with = sum(fold.train_indices.size for fold in with_embargo)
    assert total_with < total_without


def test_a_longer_label_horizon_never_yields_more_training_data() -> None:
    """Same direction-of-effect argument, applied to the purge rather than the embargo."""
    previous: list[set[int]] | None = None
    for label_horizon in (0, 1, 5, 20):
        folds = purged_kfold_indices(200, 5, label_horizon, embargo=5)
        current = [{int(i) for i in fold.train_indices} for fold in folds]
        if previous is not None:
            for stricter, looser in zip(current, previous, strict=True):
                assert stricter <= looser
        previous = current
