"""Fold each arm's completed semantic run into its audit results, so the ablation can read it.

`paradigm_audit.py` wrote `audit_results.json` while the semantic layer was still running, so its
`semantic` block holds whatever had been scored at that moment -- 49 verdicts in G1 and none in the
other five. The completed run is in `semantic_verdicts.jsonl`, one flushed line per candidate, which
is the resumable sidecar `run_paradigm_semantic.py` writes.

**This assembles data; it does not re-judge anything.** Every verdict written here is one the frozen
`src.audit.semantic` layer already produced, copied from the sidecar unchanged. No classification is
recomputed, no threshold is touched, and `src/audit/` is not read for anything but its output. The
alternative -- pointing the ablation at the sidecar -- would mean editing `run_ablation.py`, P1's
frozen apparatus, which must not be modified before a holdout evaluation under RULE 7 condition 2.

`semantic_coverage` is recomputed from the merged block so the file states its own completeness
rather than an obsolete figure.

Idempotent: rerunning it produces the same file.

Usage:
    python scripts/paradigm_merge_semantic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")


def main() -> int:
    configure_logging()
    for arm in ARMS:
        audit_path = CORPUS / arm / "audit_results.json"
        sidecar = CORPUS / arm / "semantic_verdicts.jsonl"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))

        merged = {}
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # The ablation reads confidence and rejected; name is the key. Kept verbatim otherwise.
            merged[row["name"]] = {
                "confidence": row["confidence"],
                "error": row["error"],
                "label": row["label"],
                "rejected": row["rejected"],
            }

        before = len(audit["semantic"])
        audit["semantic"] = merged
        audit["semantic_coverage"] = len(merged) / audit["n_candidates"]
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        _log.info("%-3s semantic %d -> %d verdicts, coverage %.1f%% of %d candidates",
                  arm, before, len(merged), audit["semantic_coverage"] * 100,
                  audit["n_candidates"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
