"""Diagnose the fragility predictor's instability before anyone reaches for a bigger model.

Ordered by the PI's instruction of 2026-08-02: establish whether there is predictive information in
the features at all, and whether the limitation is model bias or data variance. A higher-capacity
learner is only defensible if the simple end of the ladder is demonstrably *underfitting*.

Runs, on the deduplicated corpus with joint factor betas:

1. **Five models on identical folds** — ridge, lasso, elastic net, random forest (conservative
   defaults), and the incumbent gradient booster. Same splits, same baseline, same trimming.
2. **Five candidate causes of the collapse** — heavy tails, multicollinearity, influential
   observations, sample size, and the possibility that the relationship is simply weak.
3. **The bias-variance verdict** — in-sample against out-of-fold R^2 for every model.

Writes ``data/processed/predictor_diagnosis.json``.

Usage:
    python scripts/diagnose_predictor.py --permutations 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.diagnosis import (  # noqa: E402
    feature_collinearity,
    influence,
    learning_curve,
    permutation_test,
    target_shape,
    trimmed_scores,
)
from src.stress.predictor import MODEL_KINDS, cross_validate  # noqa: E402

_log = get_logger(__name__)

SEED = 42
DROPS = (0, 5, 10)
LEARNING_SIZES = (40, 60, 80, 100, 120)

#: The primary target is the one the PI designated (Tier 1, across paths). The charter's definition
#: and the log transform are carried alongside because Checkpoint 2.2 found they behave differently,
#: and a diagnosis of one that did not check the others would not be a diagnosis.
TARGETS = (
    ("fragility_across_paths", False),
    ("fragility_across_paths", True),
    ("fragility_across_regimes", False),
    ("fragility_across_regimes", True),
)


def _load(processed: Path) -> tuple[pl.DataFrame, list[str]]:
    """Feature table joined to targets, with knife-edge and duplicate rows removed."""
    payload = json.loads((processed / "fragility.json").read_text(encoding="utf-8"))
    targets = pl.DataFrame([
        {"name": r["name"],
         "fragility_across_regimes": r["fragility_across_regimes"],
         "fragility_across_paths": r["fragility_across_paths"]}
        for r in payload["primary"]
    ])
    table = pl.read_parquet(processed / "characteristics.parquet")
    joined = table.join(targets, on="name", how="inner").filter(
        ~pl.col("knife_edge") & ~pl.col("duplicate")
    ).sort("name")
    # `n_beta_sessions` is `n_sessions` under another name — they correlate at exactly 1.000 and
    # together drove the feature matrix's condition number to 9.9e15, which is a defect in the
    # feature table rather than a property of the data. `n_sessions` itself is retained despite
    # being a confound (bankrupt strategies have shorter series), because dropping it would hide
    # that confound rather than measure it.
    excluded = {"name", "knife_edge", "duplicate", "n_beta_sessions",
                "fragility_across_regimes", "fragility_across_paths"}
    columns = [
        c for c in joined.columns
        if c not in excluded and not c.endswith("_n")
        and not c.startswith("uni_") and joined[c].dtype.is_numeric()
    ]
    return joined, columns


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg.paths.data_processed
    joined, columns = _load(processed)
    matrix = joined.select(columns).to_numpy().astype(float)
    names = joined["name"].to_list()
    _log.info("%d strategies after removing knife-edge and duplicates, %d features",
              joined.height, len(columns))

    report: dict[str, object] = {
        "n_rows": joined.height, "n_features": len(columns), "features": columns,
        "seed": SEED, "permutations": args.permutations,
        "collinearity": feature_collinearity(matrix, columns),
        "targets": {},
    }
    _log.info("feature matrix: condition number %.1f, worst pair %s (|r| = %.3f)",
              report["collinearity"]["condition_number"],      # type: ignore[index]
              report["collinearity"]["worst_pair"],            # type: ignore[index]
              report["collinearity"]["max_abs_pairwise_correlation"])  # type: ignore[index]

    started = time.perf_counter()
    with RunManifest(cfg, script="diagnose_predictor.py") as run:
        for target_name, use_log in TARGETS:
            raw = joined[target_name].to_numpy().astype(float)
            values = np.log1p(np.maximum(raw, 0.0)) if use_log else raw
            label = f"{target_name}[{'log1p' if use_log else 'raw'}]"
            _log.info("=== %s ===", label)

            entry: dict[str, object] = {"shape": target_shape(values), "models": {}}
            _log.info("  target: skew %+.2f  excess kurtosis %+.1f  top-5 rows own %.1f%% of "
                      "the variance",
                      entry["shape"]["skewness"],           # type: ignore[index]
                      entry["shape"]["excess_kurtosis"],    # type: ignore[index]
                      100 * entry["shape"]["variance_share_top_5"])  # type: ignore[index]

            for kind in MODEL_KINDS:
                result, predictions = cross_validate(
                    matrix, values, names, target_name=label, seed=SEED, kind=kind
                )
                record = dict(result.as_dict())
                record["trimmed"] = trimmed_scores(
                    values[np.isfinite(values)], predictions, DROPS
                )
                entry["models"][kind] = record          # type: ignore[index]
                trimmed = record["trimmed"]
                _log.info(
                    "  %-18s R2 %+.3f (train %+.3f)  rho %+.3f  MAE %.3f/%.3f  "
                    "R2 drop5 %+.3f  drop10 %+.3f",
                    kind, record["r2_model"], record["r2_train"], record["spearman"],
                    record["mae_model"], record["mae_baseline"],
                    trimmed["5"]["r2"], trimmed["10"]["r2"],
                )

            # The diagnostics below are run on the best model by out-of-fold Spearman, since a
            # diagnosis of a model that never ranked anything would say nothing useful.
            best = max(
                entry["models"],                                    # type: ignore[arg-type]
                key=lambda k: entry["models"][k]["spearman"],       # type: ignore[index]
            )
            entry["best_by_spearman"] = best
            entry["influence"] = influence(matrix, values, names, kind=best, seed=SEED)
            entry["learning_curve"] = learning_curve(
                matrix, values, names, sizes=LEARNING_SIZES, kind=best, seed=SEED
            )
            entry["permutation_test"] = permutation_test(
                matrix, values, names, kind=best, seed=SEED, repeats=args.permutations
            )
            perm = entry["permutation_test"]
            _log.info("  best by rho: %s", best)
            _log.info("  influence: largest single-row delta R2 = %+.3f",
                      entry["influence"]["largest_single_delta"])   # type: ignore[index]
            _log.info("  learning curve rho: %s", "  ".join(
                f"n={int(p['n'])}: {p['spearman_mean']:+.3f}"
                for p in entry["learning_curve"]                    # type: ignore[union-attr]
            ))
            _log.info("  permutation: observed rho %+.3f vs null |rho| p95 %.3f -> p = %.4f",
                      perm["kind_spearman"], perm["null_abs_spearman_p95"],
                      perm["p_value_spearman"])
            report["targets"][label] = entry                        # type: ignore[index]

        out = processed / "predictor_diagnosis.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        run.note("output", str(out))
        run.note("n_rows", joined.height)

    _log.info("done in %.1f min -> %s", (time.perf_counter() - started) / 60, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
