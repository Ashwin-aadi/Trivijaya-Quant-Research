"""Why the fragility predictor is unstable: five candidate causes, separated by measurement.

Checkpoint 2.2 reported an out-of-fold R^2 of +0.410 that fell to -0.294 when five of 125 rows were
removed. A collapse that severe has several possible explanations, and they call for opposite
responses — so guessing between them is worse than useless. This module tests each one.

``target_shape``
    **Heavy tails.** Skewness, kurtosis, and how much of the total sum of squares the largest few
    rows own. If three rows own most of the variance, R^2 is a statement about those three rows.

``feature_collinearity``
    **Multicollinearity.** Condition number and pairwise correlation of the feature matrix. Severe
    collinearity destabilises coefficients; it does not usually destabilise a tree ensemble's
    predictions, so a large number here explains a linear model's instability and not a booster's.

``influence``
    **Influential observations.** Each row is dropped from training *and* scoring, the model refit,
    and the change in out-of-fold R^2 recorded. This is the direct measurement — leverage statistics
    from linear theory do not transfer to a boosted ensemble.

``learning_curve``
    **Sample size.** Scores at increasing subsample sizes. A curve still climbing at the full sample
    says more strategies would help; a flat one says they would not.

``permutation_test``
    **Whether there is any relationship at all.** The target is shuffled many times and the whole
    cross-validation rerun, giving the distribution of scores obtainable from features that cannot
    possibly be informative. The real score is then a percentile of that null, which is the only
    honest way to say whether +0.213 means anything on 125 rows.

Together these also answer the question that decides whether more model capacity could help: the
gap between in-sample and out-of-fold R^2. A model that fits its own training fold poorly is limited
by bias and a larger model might help; one that fits training data well and generalises badly is
limited by variance, and a larger model will make it worse.
"""

from __future__ import annotations

import numpy as np

from src.stress.predictor import N_FOLDS, _r2, assign_folds, cross_validate, spearman


def target_shape(target: np.ndarray) -> dict[str, float]:
    """Distributional shape of the target, and how concentrated its variance is."""
    values = target[np.isfinite(target)]
    centred = values - values.mean()
    variance = float(np.square(centred).mean())
    deviation = float(np.sqrt(variance))
    squared = np.square(centred)
    ordered = np.sort(squared)[::-1]
    total = float(squared.sum())
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "skewness": float((centred**3).mean() / deviation**3) if deviation > 0 else float("nan"),
        "excess_kurtosis": (
            float((centred**4).mean() / variance**2 - 3.0) if variance > 0 else float("nan")
        ),
        # The share of total squared deviation owned by the largest rows. R^2 is a ratio of sums of
        # squares, so this is literally the share of the metric those rows control.
        "variance_share_top_1": float(ordered[:1].sum() / total) if total > 0 else float("nan"),
        "variance_share_top_5": float(ordered[:5].sum() / total) if total > 0 else float("nan"),
        "variance_share_top_10": float(ordered[:10].sum() / total) if total > 0 else float("nan"),
    }


def feature_collinearity(features: np.ndarray, columns: list[str]) -> dict[str, object]:
    """Condition number and the worst-correlated feature pair, on the standardised matrix.

    Missing values are median-filled first, matching what the linear pipelines do, so the number
    describes the matrix the models actually see rather than an idealised complete one.
    """
    filled = features.copy()
    for position in range(filled.shape[1]):
        column = filled[:, position]
        fill = float(np.nanmedian(column)) if np.isfinite(column).any() else 0.0
        column[~np.isfinite(column)] = fill
    spread = filled.std(axis=0, ddof=1)
    usable = spread > 0
    standardised = (filled[:, usable] - filled[:, usable].mean(axis=0)) / spread[usable]
    singular = np.linalg.svd(standardised, compute_uv=False)
    correlation = np.corrcoef(standardised, rowvar=False)
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    worst = int(np.argmax(np.abs(np.where(mask, correlation, 0.0))))
    row, column = divmod(worst, correlation.shape[0])
    kept = [name for name, keep in zip(columns, usable, strict=True) if keep]
    return {
        "n_features": int(standardised.shape[1]),
        "n_constant_features_dropped": int((~usable).sum()),
        "condition_number": float(singular.max() / singular.min()),
        "max_abs_pairwise_correlation": float(np.abs(correlation[mask]).max()),
        "mean_abs_pairwise_correlation": float(np.abs(correlation[mask]).mean()),
        "worst_pair": [kept[row], kept[column]],
    }


def influence(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    kind: str = "gradient_boosting",
    seed: int = 42,
    top: int = 10,
) -> dict[str, object]:
    """Refit without each row in turn; report the rows whose removal moves the score most.

    Costs ``n * N_FOLDS`` fits. At ~110 rows and a shallow booster that is seconds, and it is the
    only way to identify influential observations for a model with no closed form.

    **The fold assignment is fixed once and reused.** Letting ``KFold`` re-split each time would
    change which fold every *surviving* row sits in, so the measured change would be dominated by
    reshuffling rather than by the row that was removed. That defect produced a ranking in which an
    ordinary row outranked a deliberately planted outlier, which is how it was found.
    """
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    kept_names = [name for name, keep in zip(names, finite, strict=True) if keep]
    fold_ids = assign_folds(target.shape[0], seed=seed)

    full, _ = cross_validate(
        features, target, kept_names, target_name="full", seed=seed, kind=kind,
        fold_ids=fold_ids,
    )
    deltas: list[tuple[str, float, float]] = []
    for position, name in enumerate(kept_names):
        mask = np.ones(target.shape[0], dtype=bool)
        mask[position] = False
        without, _ = cross_validate(
            features[mask], target[mask], [kept_names[i] for i in np.flatnonzero(mask)],
            target_name="loo", seed=seed, kind=kind, fold_ids=fold_ids[mask],
        )
        deltas.append((name, without.r2_model - full.r2_model, float(target[position])))

    ranked = sorted(deltas, key=lambda row: -abs(row[1]))
    return {
        "kind": kind,
        "r2_full": full.r2_model,
        "n_rows": int(target.shape[0]),
        "most_influential": [
            {"name": name, "delta_r2_when_removed": delta, "target_value": value}
            for name, delta, value in ranked[:top]
        ],
        "largest_single_delta": float(ranked[0][1]) if ranked else float("nan"),
    }


def learning_curve(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    sizes: tuple[int, ...],
    kind: str = "gradient_boosting",
    seed: int = 42,
    repeats: int = 20,
) -> list[dict[str, float]]:
    """Score against subsample size, averaged over repeated draws.

    Repeated draws matter: a single subsample of 40 rows from a heavy-tailed target may or may not
    contain the extremes, and one draw would report that accident as a trend.
    """
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    kept = [name for name, keep in zip(names, finite, strict=True) if keep]
    rng = np.random.default_rng(seed)

    out: list[dict[str, float]] = []
    for size in sizes:
        if size > target.shape[0] or size < N_FOLDS * 2:
            continue
        scores: list[float] = []
        ranks: list[float] = []
        for _ in range(repeats):
            index = rng.choice(target.shape[0], size=size, replace=False)
            result, _ = cross_validate(
                features[index], target[index], [kept[i] for i in index],
                target_name=f"n{size}", seed=seed, kind=kind,
            )
            scores.append(result.r2_model)
            ranks.append(result.spearman)
        out.append({
            "n": float(size),
            "r2_mean": float(np.mean(scores)),
            "r2_median": float(np.median(scores)),
            "spearman_mean": float(np.mean(ranks)),
            "spearman_median": float(np.median(ranks)),
            "repeats": float(repeats),
        })
    return out


def permutation_test(
    features: np.ndarray,
    target: np.ndarray,
    names: list[str],
    *,
    kind: str = "gradient_boosting",
    seed: int = 42,
    repeats: int = 200,
) -> dict[str, float]:
    """Rerun the whole cross-validation on shuffled targets to get the null distribution.

    The p-value is the fraction of shuffles scoring at least as well as the real fit, with the
    conventional ``+1`` in numerator and denominator so a p-value of exactly zero is never claimed
    from a finite number of shuffles.
    """
    finite = np.isfinite(target)
    features, target = features[finite], target[finite]
    kept = [name for name, keep in zip(names, finite, strict=True) if keep]

    observed, _ = cross_validate(
        features, target, kept, target_name="observed", seed=seed, kind=kind
    )
    rng = np.random.default_rng(seed)
    null_r2: list[float] = []
    null_rho: list[float] = []
    for _ in range(repeats):
        shuffled = rng.permutation(target)
        result, _ = cross_validate(
            features, shuffled, kept, target_name="null", seed=seed, kind=kind
        )
        null_r2.append(result.r2_model)
        null_rho.append(abs(result.spearman))

    null_r2_array = np.array(null_r2)
    null_rho_array = np.array(null_rho)
    return {
        "kind_r2": observed.r2_model,
        "kind_spearman": observed.spearman,
        "repeats": float(repeats),
        "null_r2_mean": float(null_r2_array.mean()),
        "null_r2_p95": float(np.percentile(null_r2_array, 95)),
        "p_value_r2": float((1 + (null_r2_array >= observed.r2_model).sum()) / (1 + repeats)),
        "null_abs_spearman_mean": float(null_rho_array.mean()),
        "null_abs_spearman_p95": float(np.percentile(null_rho_array, 95)),
        "p_value_spearman": float(
            (1 + (null_rho_array >= abs(observed.spearman)).sum()) / (1 + repeats)
        ),
    }


def trimmed_scores(
    target: np.ndarray, predictions: np.ndarray, drops: tuple[int, ...]
) -> dict[str, dict[str, float]]:
    """Scores after removing the k largest targets from *scoring only*, for each k.

    The model is not refitted. The question is whether the score already earned survives the removal
    of the rows dominating the sum of squares, which is not the same question as whether a model
    trained without them would do better.
    """
    finite = np.isfinite(target) & np.isfinite(predictions)
    target, predictions = target[finite], predictions[finite]
    order = np.argsort(-target)
    out: dict[str, dict[str, float]] = {}
    for k in drops:
        keep = np.ones(target.shape[0], dtype=bool)
        keep[order[:k]] = False
        reference = np.full(int(keep.sum()), float(target[keep].mean()))
        out[str(k)] = {
            "r2": _r2(target[keep], predictions[keep], reference),
            "spearman": spearman(target[keep], predictions[keep]),
            "mae": float(np.abs(target[keep] - predictions[keep]).mean()),
            "n": float(keep.sum()),
        }
    return out
