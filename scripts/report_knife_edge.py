"""What the knife-edge exclusion costs, stated as numbers rather than as an absence.

Thirty-one strategies are excluded from the primary fragility statistics and from the predictor's
training set because their Sharpe ratio is not a stable function of their inputs: reconstructing the
identical price panel by a different route moves it, in one case from -0.62 to -3.14. The PI ruled
on 2026-08-02 that they are excluded consistently — including the one standard factor among them —
and reported separately rather than discarded.

An exclusion that is never quantified is indistinguishable from a filter chosen to improve a result.
This script therefore reports the excluded population beside the retained one on every quantity the
retained population is judged on, so a reader can see whether the removal moved the answer.

Writes ``benchmarks/regimestress/knife_edge_stability.json``.

Usage:
    python scripts/report_knife_edge.py
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
from src.stress.predictor import spearman  # noqa: E402

_log = get_logger(__name__)

OUT = Path("benchmarks/regimestress/knife_edge_stability.json")

#: Quantities compared between the two populations. Chosen before the numbers were seen: these are
#: the features the predictor is given plus the two targets, so the comparison cannot be curated
#: after the fact to the ones that happen to agree.
COMPARED = (
    "fragility_across_regimes", "fragility_across_paths",
    "mean_turnover", "mean_holding_period", "effective_holdings", "book_similarity_21d",
)


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0.0, "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n": float(finite.size),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed

    fragility = json.loads((processed / "fragility.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        Path("benchmarks/regimestress/knife_edge.json").read_text(encoding="utf-8")
    )
    targets = pl.DataFrame([
        {"name": r["name"], "knife_edge": r["knife_edge"],
         "fragility_across_regimes": r["fragility_across_regimes"],
         "fragility_across_paths": r["fragility_across_paths"]}
        for r in fragility["primary"] + fragility["knife_edge_excluded"]
    ])
    features = pl.read_parquet(processed / "characteristics.parquet").drop("knife_edge")
    joined = targets.join(features, on="name", how="inner")

    retained = joined.filter(~pl.col("knife_edge"))
    excluded = joined.filter(pl.col("knife_edge"))

    comparison = {
        column: {
            "retained": _summary(retained[column].to_numpy().astype(float)),
            "excluded": _summary(excluded[column].to_numpy().astype(float)),
        }
        for column in COMPARED
    }

    # Does excluding them change the ranking of the rest? It cannot, arithmetically — but the
    # question a reader will ask is whether the excluded group is *different*, and the answer is the
    # comparison above plus the swing magnitudes below.
    swings = np.array([r["abs_sharpe_swing"] for r in frozen["knife_edge"]], dtype=float)

    payload = {
        "purpose": (
            "The cost of the knife-edge exclusion, quantified. These 31 strategies are excluded "
            "from primary fragility statistics and from the Phase 2.2 predictor training set "
            "because their Sharpe ratio is not stable under a ~9e-15 relative change in their "
            "inputs, per the PI ruling of 2026-08-02. They are reported here, not discarded."
        ),
        "n_excluded": int(excluded.height),
        "n_retained": int(retained.height),
        "n_standard_factors_excluded": frozen["n_standard_factors_affected"],
        "standard_factors_excluded": frozen["standard_factors_affected"],
        "sharpe_swing": {
            "median": float(np.median(swings)),
            "max": float(swings.max()),
            "min": float(swings.min()),
            "n_above_0_5": int((swings > 0.5).sum()),
        },
        "comparison": comparison,
        "agreement_between_definitions_within_excluded": spearman(
            excluded["fragility_across_regimes"].to_numpy().astype(float),
            excluded["fragility_across_paths"].to_numpy().astype(float),
        ),
        "agreement_between_definitions_within_retained": spearman(
            retained["fragility_across_regimes"].to_numpy().astype(float),
            retained["fragility_across_paths"].to_numpy().astype(float),
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _log.info("knife-edge: %d excluded, %d retained", excluded.height, retained.height)
    _log.info("  %-26s %18s   %18s", "", "retained (median)", "excluded (median)")
    for column, values in comparison.items():
        _log.info(
            "  %-26s %10.3f (n=%3d)   %10.3f (n=%3d)",
            column, values["retained"]["median"], int(values["retained"]["n"]),
            values["excluded"]["median"], int(values["excluded"]["n"]),
        )
    _log.info(
        "  sharpe swing: median %.4f, max %.4f, %d above 0.5",
        payload["sharpe_swing"]["median"], payload["sharpe_swing"]["max"],
        payload["sharpe_swing"]["n_above_0_5"],
    )
    _log.info("  written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
