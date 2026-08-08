"""Do the paradigms produce different strategies from each other, or the same ones?

**Exploratory. Not pre-registered.** R(G) as specified in PREREGISTRATION.md measures duplication
*within* an arm, which answers "does this paradigm repeat itself". It cannot answer "does this
paradigm produce anything the others do not", and the two are different questions with different
consequences for RQ3.

The question surfaced from a tell rather than a hypothesis: the stratified capacity table showed the
same binding capacity to the paisa in several arms at once -- 3.85 crore in five of the six, 0.19
crore in four. Identical capacity is suggestive but not proof, since two different strategies could
coincide. This settles it on the quantity that actually defines a duplicate, by pooling every arm's
traded strategies and running P2's union-find over realised net returns unchanged -- the identical
rule, tolerance and code path `paradigm_redundancy.py` uses within arms.

**Why it matters.** If a large share of clusters span arms, then the paradigm axis is producing less
distinct output than within-arm R(G) implies, and any claim that a paradigm explores a different
region of strategy space is weakened in proportion. That is a finding about the paradigms, not about
the measurement.

Writes `benchmarks/generationbench/cross_arm_duplicates.json`.

Usage:
    python scripts/paradigm_cross_arm_duplicates.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from paradigm_redundancy import EXACT_TOLERANCE, _clusters  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
OUT = Path("benchmarks/generationbench/cross_arm_duplicates.json")


def pooled_series() -> tuple[list[str], np.ndarray]:
    """Arm-qualified names and the aligned net-return matrix for every traded strategy, all arms.

    Names are qualified as ``<arm>/<candidate>`` because candidate numbering restarts per arm, so
    the bare name is ambiguous once the arms are pooled.
    """
    series: dict[str, np.ndarray] = {}
    for arm in ARMS:
        results = json.loads(
            (CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
        for row in results:
            if row["outcome"] != "evaluated" or not (row.get("mean_turnover") or 0) > 0:
                continue
            path = row.get("returns_path")
            if not path or not Path(path).exists():
                continue
            series[f"{arm}/{row['name']}"] = (
                pl.read_parquet(path).sort("session_date")["return"].to_numpy())

    if not series:
        return [], np.empty((0, 0))
    # Same rule as the within-arm run: a strategy ruined early has a shorter series and cannot be a
    # duplicate of a complete one, so it is reported as uncompared rather than dropped.
    full = max(len(v) for v in series.values())
    names = sorted(n for n, v in series.items() if len(v) == full)
    return names, np.vstack([series[n] for n in names])


def main() -> int:
    configure_logging()
    names, matrix = pooled_series()
    _log.info("pooled %d traded strategies across %d arms at full length", len(names), len(ARMS))

    clusters = _clusters(names, matrix)
    spanning = [c for c in clusters if len({n.split("/")[0] for n in c}) > 1]
    in_spanning = sorted({n for c in spanning for n in c})

    _log.info("%d clusters pooled, %d of them spanning more than one arm", len(clusters),
              len(spanning))
    _log.info("%d of %d traded strategies (%.1f%%) sit in a cross-arm cluster",
              len(in_spanning), len(names), len(in_spanning) / len(names) * 100)

    per_arm = Counter(n.split("/")[0] for n in in_spanning)
    traded_per_arm = Counter(n.split("/")[0] for n in names)
    for arm in ARMS:
        share = per_arm[arm] / traded_per_arm[arm] * 100 if traded_per_arm[arm] else float("nan")
        _log.info("  %-3s %3d of %3d traded (%.1f%%) duplicated in another arm",
                  arm, per_arm[arm], traded_per_arm[arm], share)

    widest = max(spanning, key=lambda c: len({n.split("/")[0] for n in c}), default=[])
    if widest:
        _log.info("widest cluster spans %d arms with %d members",
                  len({n.split("/")[0] for n in widest}), len(widest))

    OUT.write_text(json.dumps({
        "status": "EXPLORATORY -- not pre-registered; prompted by ties in the capacity table",
        "rule": ("P2's union-find over realised net returns, imported unchanged from "
                 "paradigm_redundancy; identical tolerance and code path as the within-arm run"),
        "exact_tolerance": EXACT_TOLERANCE,
        "n_traded_pooled": len(names),
        "n_clusters_pooled": len(clusters),
        "n_clusters_spanning_arms": len(spanning),
        "n_strategies_in_cross_arm_cluster": len(in_spanning),
        "share_in_cross_arm_cluster": len(in_spanning) / len(names) if names else float("nan"),
        "per_arm_traded": dict(traded_per_arm),
        "per_arm_in_cross_arm_cluster": dict(per_arm),
        "spanning_clusters": spanning,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
