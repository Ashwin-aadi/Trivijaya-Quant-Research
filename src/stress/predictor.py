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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    #: Which model produced this. Part of the record so a table of results cannot lose track.
    kind: str = "gradient_boosting"
    #: Mean in-sample R^2 over the training folds. Compared against ``r2_model``: a small gap with
    #: both low means the model class is too weak (bias); a large gap means it is fitting noise
    #: (variance). This is the measurement that decides whether more capacity could ever help.
    r2_train: float = float("nan")
    #: Permutation importance, mean over folds. Feature name -> increase in out-of-fold error.
    importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "r2_train": self.r2_train,
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


#: Every model that gets compared, in increasing order of capacity. The point of the ladder is that
#: a high-capacity learner is only worth reaching for if the simple end of it is *underfitting*; if
#: a ridge and a boosted ensemble score the same, the ceiling is the data, not the model class.
#:
#: Regularisation strengths are the library defaults except where noted. They are not tuned, and
#: deliberately so: tuning a hyperparameter on the same folds the score is read from is how a null
#: result is converted into a positive one.
MODEL_KINDS = ("ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting")


def _model(seed: int, kind: str = "gradient_boosting") -> Pipeline | HistGradientBoostingRegressor:
    """One model by name, wrapped so that fitting cannot see outside its training fold.

    The linear models sit behind a median imputer and a standardiser, both fitted *inside* the
    pipeline and therefore inside the fold. Standardising once over the whole matrix would be a
    full-sample transform applied to test data — the exact violation Project 1 exists to detect —
    and the features here differ by orders of magnitude, so it cannot simply be skipped.

    The two tree ensembles need neither: ``HistGradientBoostingRegressor`` handles missing values
    natively and both are invariant to monotone rescaling of a feature.
    """
    if kind == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=MAX_ITER, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, random_state=seed,
        )
    if kind == "random_forest":
        # Conservative: shallow trees and a floor on leaf size, so it cannot memorise 125 rows.
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500, max_depth=5, min_samples_leaf=5,
                random_state=seed, n_jobs=-1,
            )),
        ])
    estimators = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        # 200k iterations, not the default 1k: coordinate descent does not converge on the raw
        # fragility target, whose scale is three orders of magnitude above the standardised
        # features. A non-converged fit is not a fair entry in a model comparison.
        "lasso": Lasso(alpha=0.01, random_state=seed, max_iter=200_000),
        "elastic_net": ElasticNet(
            alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=200_000
        ),
    }
    if kind not in estimators:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimators[kind]),
    ])


def cross_validate(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    target_name: str,
    seed: int = 42,
    kind: str = "gradient_boosting",
    fold_ids: np.ndarray | None = None,
) -> tuple[PredictorResult, np.ndarray]:
    """Out-of-fold predictions and their honest scores. Returns the result and the predictions.

    ``kind`` selects the model; the fold split depends only on ``seed`` and the row count, so every
    model in :data:`MODEL_KINDS` is scored on identical splits and the comparison between them is
    not confounded by which rows each happened to be tested on.

    ``fold_ids`` overrides the split with a caller-supplied assignment, one fold index per row. It
    exists for leave-one-out influence: removing a row changes the row count, which reshuffles a
    ``KFold`` split and moves every *other* row to a different fold. The resulting score change
    would then be mostly resampling noise rather than the influence of the row removed.
    """
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    if fold_ids is not None:
        fold_ids = fold_ids[finite]
    predictions = np.full(target.shape[0], np.nan)
    baseline = np.full(target.shape[0], np.nan)
    train_scores: list[float] = []

    splits = (
        [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f))
         for f in np.unique(fold_ids)]
        if fold_ids is not None
        else list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(features))
    )
    for train_index, test_index in splits:
        if test_index.size == 0 or train_index.size < 2:
            continue
        model = _model(seed, kind)
        model.fit(features[train_index], target[train_index])
        predictions[test_index] = model.predict(features[test_index])
        # In-sample fit on the same fold. The gap between this and the out-of-fold score is what
        # separates a model that is underfitting from one that is memorising.
        fitted = model.predict(features[train_index])
        reference = np.full(train_index.shape[0], float(target[train_index].mean()))
        train_scores.append(_r2(target[train_index], fitted, reference))
        # The baseline sees only the training fold, exactly as the model does.
        baseline[test_index] = float(target[train_index].mean())

    result = PredictorResult(
        kind=kind,
        r2_train=float(np.mean(train_scores)),
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


def assign_folds(n_rows: int, *, seed: int = 42) -> np.ndarray:
    """Fold index per row, from the same ``KFold`` the scorer uses by default.

    Materialised so a caller can hold the assignment fixed across refits — see ``fold_ids``.
    """
    ids = np.empty(n_rows, dtype=int)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros((n_rows, 1)))):
        ids[test_index] = fold
    return ids


def permutation_importance(
    features: np.ndarray,
    target: np.ndarray,
    columns: list[str],
    *,
    seed: int = 42,
    repeats: int = 10,
    kind: str = "gradient_boosting",
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
        model = _model(seed, kind)
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
