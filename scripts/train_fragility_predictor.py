"""Train and honestly score the fragility predictor — the Phase 2.2 ML contribution.

Joins the feature table from ``scripts/build_characteristics.py`` to the targets from
``scripts/build_fragility.py`` and cross-validates a gradient-boosted regressor against a
per-fold mean baseline.

Two targets are scored, both reported:

* ``fragility_across_paths`` — **primary**, measured by the Tier 1 counterfactual run. The PI's
  Checkpoint 2.1 ruling: "Train the predictor using Tier 1. Tier 1 is the ground-truth stress
  measurement."
* ``fragility_across_regimes`` — the charter's definition, computed on the realised series.

Knife-edge strategies are excluded from training per the same ruling and scored separately, so
their exclusion is visible as a number rather than as an absence. **Exact duplicates are also
excluded**, per the PI ruling of 2026-08-02: the corpus contains 11 clusters of strategies with
identical realised return series, and leaving them in lets a row sit in a training fold while its
twin is scored in the test fold. On the primary target that leakage was worth **+0.238 of R^2**
(+0.262 with duplicates against +0.024 without, identical features and folds).

Two variants of each target are run because fragility is a heavy-tailed ratio and a log transform is
a real modelling fork, not a detail: the raw target and ``log1p`` of it. Both are reported. Neither
is chosen here.

Usage:
    python scripts/train_fragility_predictor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.predictor import (  # noqa: E402
    _r2,
    cross_validate,
    permutation_importance,
    spearman,
)

_log = get_logger(__name__)

SEED = 42
TARGETS = ("fragility_across_paths", "fragility_across_regimes")

#: Fragility is a ratio with a mean in the denominator and its distribution is extremely
#: right-skewed — median 0.36 against a maximum of 46. An R^2 computed on such a target can be
#: carried entirely by a handful of rows, so it is recomputed with the largest few removed. If the
#: score collapses, the model is fitting the tail rather than the population, and reporting the
#: headline figure alone would overstate what it has learned.
DROP_TOP = (1, 2, 5, 10)

#: Columns that are bookkeeping rather than characteristics. The ``_n`` columns are the sample sizes
#: behind the similarity features and are near-duplicates of ``n_sessions``; they stay in the table
#: for reporting but are not offered to the model as evidence.
NOT_FEATURES = {"name", "knife_edge", "duplicate", "n_beta_sessions"}


def _feature_columns(table: pl.DataFrame) -> list[str]:
    return [
        c for c in table.columns
        if c not in NOT_FEATURES and not c.endswith("_n")
        and not c.startswith("uni_") and table[c].dtype.is_numeric()
    ]


def _tail_sensitivity(target: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Out-of-fold R^2 after removing the k largest targets, for each k in :data:`DROP_TOP`.

    The predictions are not refitted: the question is whether the score the model already earned
    survives the removal of the rows that dominate the sum of squares, not whether a different
    model trained without them would do better.
    """
    order = np.argsort(-target)
    out: dict[str, float] = {}
    for k in DROP_TOP:
        keep = np.ones(target.shape[0], dtype=bool)
        keep[order[:k]] = False
        reference = np.full(int(keep.sum()), float(target[keep].mean()))
        out[str(k)] = _r2(target[keep], predictions[keep], reference)
    return out


def _targets(processed: Path) -> pl.DataFrame:
    payload = json.loads((processed / "fragility.json").read_text(encoding="utf-8"))
    rows = payload["primary"] + payload["knife_edge_excluded"]
    return pl.DataFrame([
        {
            "name": r["name"],
            "knife_edge": r["knife_edge"],
            "fragility_across_regimes": r["fragility_across_regimes"],
            "fragility_across_paths": r["fragility_across_paths"],
            "mean_is_near_zero": r["mean_is_near_zero"],
        }
        for r in rows
    ])


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed

    features_table = pl.read_parquet(processed / "characteristics.parquet")
    joined = features_table.join(
        _targets(processed).drop("knife_edge"), on="name", how="inner"
    ).sort("name")
    training = joined.filter(~pl.col("knife_edge") & ~pl.col("duplicate"))
    held_out = joined.filter(pl.col("knife_edge") | pl.col("duplicate"))
    columns = _feature_columns(training.drop(list(TARGETS) + ["mean_is_near_zero"]))
    _log.info(
        "%d strategies (%d excluded as knife-edge or duplicate), %d features",
        training.height, held_out.height, len(columns),
    )

    matrix = training.select(columns).to_numpy().astype(float)
    report: dict[str, object] = {
        "n_training": training.height,
        "n_excluded_knife_edge": int(joined["knife_edge"].sum()),
        "n_excluded_duplicate": int((joined["duplicate"] & ~joined["knife_edge"]).sum()),
        "n_features": len(columns),
        "features": columns,
        "seed": SEED,
        "results": [],
    }

    with RunManifest(cfg, script="train_fragility_predictor.py") as run:
        for target_name in TARGETS:
            raw = training[target_name].to_numpy().astype(float)
            for transform, values in (("raw", raw), ("log1p", np.log1p(np.maximum(raw, 0.0)))):
                label = f"{target_name}[{transform}]"
                result, predictions = cross_validate(
                    matrix, values, training["name"].to_list(),
                    target_name=label, seed=SEED,
                )
                record = dict(result.as_dict())
                # Rank agreement against the *raw* target either way: a log transform must not be
                # allowed to change what is being ranked.
                finite = np.isfinite(values)
                record["spearman_vs_raw_target"] = spearman(raw[finite], predictions)
                record["r2_dropping_largest"] = _tail_sensitivity(values[finite], predictions)
                if transform == "raw":
                    record["importance"] = permutation_importance(
                        matrix, values, columns, seed=SEED
                    )
                report["results"].append(record)  # type: ignore[union-attr]
                _log.info(
                    "%-42s  R2 %+.3f vs baseline 0.000   rho %+.3f   MAE %.3f vs %.3f   n=%d",
                    label, record["r2_model"], record["spearman"],
                    record["mae_model"], record["mae_baseline"], record["n_rows"],
                )
                _log.info(
                    "%-42s  R2 dropping largest: %s", "",
                    "  ".join(f"-{k}: {v:+.3f}" for k, v in record["r2_dropping_largest"].items()),
                )

        out = processed / "fragility_predictor.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        run.note("output", str(out))
        run.note("n_training", training.height)

    primary = next(
        r for r in report["results"]  # type: ignore[union-attr]
        if r["target"] == "fragility_across_paths[raw]"
    )
    ranked = sorted(primary["importance"].items(), key=lambda kv: -kv[1])
    _log.info("top permutation importances (increase in out-of-fold MAE when shuffled):")
    for column, value in ranked[:8]:
        _log.info("  %-32s %+.4f", column, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
