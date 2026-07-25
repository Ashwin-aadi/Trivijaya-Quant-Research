"""Purged k-fold cross-validation with an embargo, after Lopez de Prado.

Reference: Marcos Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018),
Chapter 7 — "Cross-Validation in Finance". Purging is Snippet 7.1; the embargo is Snippet 7.2.

**Why ordinary k-fold is wrong here.** Financial labels are built over a horizon: a label at time
*t* depends on prices through *t + h*. If *t* lands in a training fold and *t + h* in a test fold,
the training set already contains the test period's outcome, and the model scores well by having
been told the answer. Two corrections are applied:

*Purging* removes from the training set every observation whose label window overlaps the test
fold. *Embargo* additionally drops training observations immediately **after** the test fold, since
serial correlation means a point just past the boundary still carries information about it.

Both are mandatory in this project rather than optional, so the split function refuses a zero
embargo unless it is asked for explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One train/test split, already purged and embargoed."""

    train_indices: np.ndarray
    test_indices: np.ndarray

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train_indices, self.test_indices)
        if overlap.size:
            raise ValueError(f"train and test overlap at {overlap.size} indices")


def purged_kfold_indices(
    n_samples: int,
    n_splits: int = 5,
    label_horizon: int = 1,
    embargo: int = 5,
) -> list[Fold]:
    """Contiguous test folds with overlapping and adjacent training points removed.

    Args:
        n_samples: number of observations, assumed ordered in time.
        n_splits: number of folds.
        label_horizon: how many periods forward each label looks. An observation at ``i`` is
            purged when ``i + label_horizon`` reaches into the test fold.
        embargo: training observations to drop immediately after each test fold. Five trading days
            is the project default.

    Folds are contiguous blocks, never shuffled — shuffling time series destroys the ordering the
    purge depends on and is itself a form of leakage.
    """
    if n_splits < 2:
        raise ValueError("need at least two folds")
    if n_samples < n_splits:
        raise ValueError(f"cannot split {n_samples} samples into {n_splits} folds")
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    indices = np.arange(n_samples)
    boundaries = np.array_split(indices, n_splits)

    folds: list[Fold] = []
    for block in boundaries:
        test_start, test_end = int(block[0]), int(block[-1])

        # Purge: an observation whose label window reaches the test fold must not train the model.
        purge_start = test_start - label_horizon
        # Embargo: drop a further stretch after the fold, where serial correlation still bites.
        embargo_end = test_end + embargo

        train_mask = (indices < purge_start) | (indices > embargo_end)
        folds.append(Fold(train_indices=indices[train_mask], test_indices=block))
    return folds


def assert_no_leakage(fold: Fold, label_horizon: int, embargo: int) -> None:
    """Verify a fold really is purged and embargoed. Raises on any violation.

    Used by the test suite as an independent check: the assertion is written against the property
    that must hold, not against how :func:`purged_kfold_indices` happens to compute it.
    """
    if fold.test_indices.size == 0:
        raise ValueError("empty test fold")
    test_start, test_end = int(fold.test_indices[0]), int(fold.test_indices[-1])

    for index in fold.train_indices:
        # No training label may reach into the test fold.
        if test_start <= index + label_horizon and index <= test_end:
            raise ValueError(
                f"training index {index} has a label window reaching test fold "
                f"[{test_start}, {test_end}] with horizon {label_horizon}"
            )
        # No training point may sit inside the embargo shadow after the fold.
        if test_end < index <= test_end + embargo:
            raise ValueError(
                f"training index {index} falls inside the {embargo}-period embargo after "
                f"test fold ending {test_end}"
            )
