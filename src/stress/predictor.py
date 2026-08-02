"""Can fragility be predicted from what a strategy *is*, without running the stress suite?

The Phase 2.2 research question, stated so it can fail: a gradient-boosted regressor maps strategy
characteristics to a fragility score, and is scored against the only baseline that matters — the
mean of the training fold. A model that cannot beat "guess the average" has learned nothing,
however respectable its R^2 looks in isolation.

Three deliberate choices, each of which could have been made to flatter the result:

* **The baseline is fitted per fold**, not once on the whole sample. A baseline that has seen the
  test fold's mean is not a baseline.
* **R^2 is reported out-of-fold**, computed against the *training* mean rather than the test mean.
  Using the test mean makes the denominator smaller on hard folds and inflates the score.
* **Spearman is reported alongside R^2.** Fragility is heavy-tailed; a model can rank strategies
  usefully while missing the extremes badly, and one number would hide which is happening.

Cross-validation here is plain k-fold over strategies. There is no time axis to purge: each row is
a whole strategy summarised over the same decade. What that does *not* remove is that all rows share
one market history, so the folds are not independent draws and the confidence intervals below are
narrower than the truth. Stated, not corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

#: Folds for cross-validation. Five over ~125 strategies leaves 25 per test fold — enough for a
#: Spearman correlation to mean something, few enough that the training folds stay usable.
N_FOLDS = 5

#: Trees are shallow and few on purpose: 125 rows and ~25 features is a regime where an
#: unconstrained booster will fit the training set perfectly and tell you nothing.
MAX_ITER = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.05


@dataclass
class PredictorResult:
    """Out-of-fold performance of one target, against the mean baseline."""

    target: str
    n_rows: int
    n_features: int
    r2_model: float
    r2_baseline: float
    spearman: float
    mae_model: float
    mae_baseline: float
    #: Permutation importance, mean over folds. Feature name -> increase in out-of-fold error.
    importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "r2_model": self.r2_model,
            "r2_baseline": self.r2_baseline,
            "spearman": self.spearman,
            "mae_model": self.mae_model,
            "mae_baseline": self.mae_baseline,
            "importance": self.importance,
        }


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed here rather than imported to avoid a scipy dependency."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ranks = [np.argsort(np.argsort(v[ok])).astype(float) for v in (a, b)]
    centred = [r - r.mean() for r in ranks]
    denom = float(np.sqrt((centred[0] ** 2).sum() * (centred[1] ** 2).sum()))
    return float((centred[0] * centred[1]).sum() / denom) if denom else float("nan")


def _r2(truth: np.ndarray, prediction: np.ndarray, reference: np.ndarray) -> float:
    """``1 - SSE/SST`` with ``SST`` about ``reference``: the training mean, not the test mean."""
    sse = float(np.square(truth - prediction).sum())
    sst = float(np.square(truth - reference).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def _model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=seed,
    )


def cross_validate(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    target_name: str,
    seed: int = 42,
) -> tuple[PredictorResult, np.ndarray]:
    """Out-of-fold predictions and their honest scores. Returns the result and the predictions."""
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    predictions = np.full(target.shape[0], np.nan)
    baseline = np.full(target.shape[0], np.nan)

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_index, test_index in folds.split(features):
        model = _model(seed)
        model.fit(features[train_index], target[train_index])
        predictions[test_index] = model.predict(features[test_index])
        # The baseline sees only the training fold, exactly as the model does.
        baseline[test_index] = float(target[train_index].mean())

    result = PredictorResult(
        target=target_name,
        n_rows=int(target.shape[0]),
        n_features=int(features.shape[1]),
        r2_model=_r2(target, predictions, baseline),
        r2_baseline=0.0,      # by construction: the baseline is the reference
        spearman=spearman(target, predictions),
        mae_model=float(np.abs(target - predictions).mean()),
        mae_baseline=float(np.abs(target - baseline).mean()),
    )
    return result, predictions


def permutation_importance(
    features: np.ndarray,
    target: np.ndarray,
    columns: list[str],
    *,
    seed: int = 42,
    repeats: int = 10,
) -> dict[str, float]:
    """Increase in out-of-fold MAE when a column is shuffled. Positive means the column carried
    information the model used.

    Permutation rather than the booster's internal split counts: split counts reward
    high-cardinality features regardless of whether they generalise, and with 125 rows that
    difference is not academic.
    """
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    rng = np.random.default_rng(seed)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(features))

    scores: dict[str, list[float]] = {c: [] for c in columns}
    for train_index, test_index in folds:
        model = _model(seed)
        model.fit(features[train_index], target[train_index])
        base = float(np.abs(target[test_index] - model.predict(features[test_index])).mean())
        for position, column in enumerate(columns):
            losses = []
            for _ in range(repeats):
                shuffled = features[test_index].copy()
                shuffled[:, position] = rng.permutation(shuffled[:, position])
                losses.append(float(np.abs(target[test_index] - model.predict(shuffled)).mean()))
            scores[column].append(float(np.mean(losses)) - base)
    return {column: float(np.mean(values)) for column, values in scores.items()}
