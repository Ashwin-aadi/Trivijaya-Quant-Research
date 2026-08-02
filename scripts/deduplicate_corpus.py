"""Find strategies that are the same strategy under a different name, before anything is reported.

Checkpoint 2.2 turned up two candidates with regime Sharpe ratios agreeing to three decimal places.
If the generated corpus contains duplicates, three things reported from it are wrong in the same
direction: the effective sample is smaller than 125, cross-validation leaks because a duplicate can
sit in the training fold while its twin is scored in the test fold, and a feature that identifies a
duplicated cluster looks predictive because it is predicting the same rows twice.

Duplication is judged on the **realised net return series**, not on source text. Two strategies with
different code that produce identical returns on every one of 1,232 sessions are the same strategy
for every purpose in this project, and two with near-identical code that diverge are not.

Two thresholds are reported, both:

* **exact** — every session agrees to within ``EXACT_TOLERANCE``. This is the primary criterion and
  the one used to build the retained set.
* **near** — Pearson correlation above ``NEAR_CORRELATION``. Reported so the choice of the strict
  criterion is visible as a number rather than an assumption; a large gap between the two counts
  would mean the strict rule is missing clusters that are duplicates in everything but rounding.

Within a duplicate cluster the alphabetically first name is retained, which is arbitrary but fixed:
choosing by any performance quantity would be selection on the outcome.

Writes ``benchmarks/regimestress/duplicates.json``.

Usage:
    python scripts/deduplicate_corpus.py
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

_log = get_logger(__name__)

OUT = Path("benchmarks/regimestress/duplicates.json")

#: Two return series count as identical if no session differs by more than this. Daily net returns
#: are of order 1e-2, so 1e-12 is ten orders of magnitude below the signal: it admits floating-point
#: reassociation and nothing else.
EXACT_TOLERANCE = 1e-12

#: Reported alongside, never used to remove anything. Correlation this high with a *non*-zero
#: difference means two strategies that move together but are not the same object.
NEAR_CORRELATION = 0.9999


def _clusters(names: list[str], matrix: np.ndarray) -> list[list[str]]:
    """Group indices whose series agree everywhere, by union-find over pairwise comparisons.

    Transitivity matters: if A equals B and B equals C, all three are one cluster even if A and C
    were never compared directly. Written explicitly rather than by thresholding a distance matrix,
    because a cluster of three would otherwise be reported as two overlapping pairs.
    """
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.max(np.abs(matrix[i] - matrix[j])) <= EXACT_TOLERANCE:
                parent[find(j)] = find(i)

    grouped: dict[int, list[str]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(find(index), []).append(name)
    return [sorted(members) for members in grouped.values() if len(members) > 1]


def _near_pairs(names: list[str], matrix: np.ndarray, exact: set[str]) -> list[dict[str, object]]:
    """Pairs correlating above the near threshold that the exact rule did *not* merge."""
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    out: list[dict[str, object]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i] in exact and names[j] in exact:
                continue
            denom = float(norms[i] * norms[j])
            if denom <= 0:
                continue
            correlation = float(centred[i] @ centred[j] / denom)
            if correlation >= NEAR_CORRELATION:
                out.append({"a": names[i], "b": names[j], "correlation": correlation,
                            "max_abs_difference": float(np.max(np.abs(matrix[i] - matrix[j])))})
    return sorted(out, key=lambda r: -float(r["correlation"]))  # type: ignore[arg-type]


def main() -> int:
    configure_logging()
    cfg = load_config()
    returns = pl.read_parquet(cfg.paths.data_processed / "real_returns.parquet")

    # Only strategies sharing the full session count are comparable elementwise. A strategy ruined
    # early has a shorter series and cannot be a duplicate of a complete one; it is retained
    # untouched and reported as uncompared rather than silently dropped from the census.
    lengths = returns.group_by("name").len()
    full = int(lengths["len"].max())
    comparable = sorted(lengths.filter(pl.col("len") == full)["name"].to_list())
    short = sorted(lengths.filter(pl.col("len") != full)["name"].to_list())

    wide = returns.filter(pl.col("name").is_in(comparable)).pivot(
        on="name", index="session_date", values="net_return"
    ).sort("session_date")
    matrix = np.column_stack([wide[name].to_numpy() for name in comparable]).T

    clusters = _clusters(comparable, matrix)
    duplicated = {name for cluster in clusters for name in cluster}
    removed = sorted(name for cluster in clusters for name in cluster[1:])
    near = _near_pairs(comparable, matrix, duplicated)

    payload = {
        "purpose": (
            "Strategies whose realised net return series are identical to within 1e-12 on every "
            "session. Within each cluster the alphabetically first name is retained and the rest "
            "are removed before feature importances and cross-validation, so a duplicate cannot "
            "sit in a training fold while its twin is scored in the test fold."
        ),
        "exact_tolerance": EXACT_TOLERANCE,
        "near_correlation_threshold": NEAR_CORRELATION,
        "n_compared": len(comparable),
        "n_uncompared_short_series": len(short),
        "uncompared_short_series": short,
        "n_clusters": len(clusters),
        "n_strategies_in_clusters": len(duplicated),
        "n_removed": len(removed),
        "removed": removed,
        "clusters": clusters,
        "n_near_duplicate_pairs_not_merged": len(near),
        "near_duplicate_pairs_not_merged": near[:50],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _log.info("%d strategies compared over %d sessions (%d short series uncompared)",
              len(comparable), full, len(short))
    _log.info("%d exact-duplicate clusters covering %d strategies; %d removed, %d retained",
              len(clusters), len(duplicated), len(removed), len(comparable) - len(removed))
    for cluster in clusters[:10]:
        _log.info("    %s", " == ".join(cluster))
    _log.info("%d near-duplicate pairs (corr >= %.4f) NOT merged by the exact rule",
              len(near), NEAR_CORRELATION)
    for pair in near[:5]:
        _log.info("    %s ~ %s   corr %.6f   max diff %.2e",
                  pair["a"], pair["b"], pair["correlation"], pair["max_abs_difference"])
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
