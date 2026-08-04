"""Train P2's fragility predictor on P2's corpus, then predict the frontier arms it never saw.

P2 reported that predicting fragility from strategy characteristics does not work: an out-of-sample
R-squared of +0.024 against a mean baseline, diagnosed as a sample-size and collinearity failure
rather than a missing feature. This script asks whether that negative result is a property of the
local corpus or of the problem, by holding the model fixed and changing the population.

**The model, the features and the seed are P2's.** ``_model`` and ``assign_folds`` are imported
from :mod:`src.stress.predictor`, the feature columns are selected by P2's own rule, and training
uses P2's exclusions --- knife-edge and duplicate strategies out. Nothing is retuned for the
frontier arms, which are pure held-out data: they contribute nothing to fitting.

**This analysis is exploratory.** It appears nowhere in the generator-validation pre-registration,
and every figure it produces must be labelled as such in the same sentence as the figure.

Usage:
    python scripts/frontier_fragility_predict.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from train_fragility_predictor import SEED, _feature_columns, _targets  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.stress.predictor import _model, spearman  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gpt", "claude", "gemini")
TARGET = "fragility_across_paths"


def _r2_against_baseline(truth: np.ndarray, prediction: np.ndarray, baseline: float) -> float:
    """R-squared against a constant baseline fixed on the *training* population.

    Scoring against the frontier arm's own mean would let the baseline see the labels being
    predicted, which flatters any model that fails to beat it. The honest reference for a model
    deployed on a new population is the number it would have predicted knowing only its training
    set.
    """
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - baseline) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def _arm_frame(arm: str, columns: list[str]) -> tuple[pl.DataFrame, np.ndarray]:
    """The arm's feature matrix in training-column order, and its measured fragility."""
    run = ROOT / "runs" / f"frontier_{arm}"
    features = pl.read_parquet(run / "characteristics.parquet")
    measured = json.loads((run / "fragility.json").read_text(encoding="utf-8"))
    keep = [n for n in features["name"].to_list() if n in measured]
    features = features.filter(pl.col("name").is_in(keep)).sort("name")
    truth = np.array(
        [measured[n]["fragility_across_paths"] for n in features["name"].to_list()], dtype=float
    )
    missing = [c for c in columns if c not in features.columns]
    for column in missing:
        features = features.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    if missing:
        _log.warning("arm %s: %d training columns absent, filled null: %s",
                     arm, len(missing), missing)
    return features, truth


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed

    table = pl.read_parquet(processed / "characteristics.parquet")
    joined = table.join(_targets(processed).drop("knife_edge"), on="name", how="inner").sort("name")
    training = joined.filter(~pl.col("knife_edge") & ~pl.col("duplicate"))
    columns = _feature_columns(
        training.drop(["fragility_across_paths", "fragility_across_regimes", "mean_is_near_zero"])
    )
    matrix = training.select(columns).to_numpy().astype(float)
    target = training[TARGET].to_numpy().astype(float)
    baseline = float(np.mean(target))
    _log.info("training on %d local strategies, %d features, seed %d",
              training.height, len(columns), SEED)
    _log.info("training-set mean fragility (the baseline predictor): %.4f", baseline)

    model = _model(SEED)
    model.fit(matrix, target)

    results: dict[str, Any] = {
        "exploratory": True,
        "note": (
            "Not pre-registered. P2 reported this predictor does not work in its own population "
            "(out-of-sample R2 +0.024); this asks only whether that null survives a change of "
            "generator."
        ),
        "target": TARGET,
        "seed": SEED,
        "n_training": training.height,
        "n_features": len(columns),
        "training_mean": baseline,
        "arms": {},
    }

    for arm in ARMS:
        features, truth = _arm_frame(arm, columns)
        prediction = model.predict(features.select(columns).to_numpy().astype(float))
        row = {
            "n": int(len(truth)),
            "r2_vs_training_mean": _r2_against_baseline(truth, prediction, baseline),
            "r2_vs_own_mean": _r2_against_baseline(truth, prediction, float(np.mean(truth))),
            "spearman": spearman(truth, prediction),
            "mae_model": float(np.mean(np.abs(truth - prediction))),
            "mae_baseline": float(np.mean(np.abs(truth - baseline))),
            "measured_median": float(np.median(truth)),
            "predicted_median": float(np.median(prediction)),
            "measured_range": [float(truth.min()), float(truth.max())],
            "predicted_range": [float(prediction.min()), float(prediction.max())],
        }
        results["arms"][arm] = row
        _log.info(
            "  %-7s n=%2d  R2(train mean) %+7.3f  R2(own mean) %+7.3f  rho %+.3f  "
            "MAE %.3f vs %.3f",
            arm, row["n"], row["r2_vs_training_mean"], row["r2_vs_own_mean"],
            row["spearman"], row["mae_model"], row["mae_baseline"],
        )

    pooled_truth = np.concatenate([
        _arm_frame(arm, columns)[1] for arm in ARMS
    ])
    pooled_pred = np.concatenate([
        model.predict(_arm_frame(arm, columns)[0].select(columns).to_numpy().astype(float))
        for arm in ARMS
    ])
    results["pooled"] = {
        "n": int(len(pooled_truth)),
        "r2_vs_training_mean": _r2_against_baseline(pooled_truth, pooled_pred, baseline),
        "spearman": spearman(pooled_truth, pooled_pred),
        "mae_model": float(np.mean(np.abs(pooled_truth - pooled_pred))),
        "mae_baseline": float(np.mean(np.abs(pooled_truth - baseline))),
    }
    _log.info("  pooled  n=%2d  R2(train mean) %+7.3f  rho %+.3f  MAE %.3f vs %.3f",
              results["pooled"]["n"], results["pooled"]["r2_vs_training_mean"],
              results["pooled"]["spearman"], results["pooled"]["mae_model"],
              results["pooled"]["mae_baseline"])

    out = processed / "frontier_fragility_prediction.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Not relative_to: the configured data path may be relative, and raising here would fail the
    # run after the artifact was already written.
    _log.info("wrote %s", out.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
