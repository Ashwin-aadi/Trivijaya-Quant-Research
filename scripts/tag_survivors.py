"""Freeze the strategies that survived every auditor layer, as the P1 output P2 consumes.

A survivor is a candidate that did three things: executed against real data, actually traded, and
was rejected by none of the three auditor layers. That is the population the lab's pipeline story
depends on — P2 asks when these break, P3 asks how much money they absorb — so it has to exist as a
concrete, labelled, reproducible set rather than as a claim in a report.

**The three layers are recorded separately for every survivor, not collapsed.** A candidate passes
because static, semantic and statistical each passed it, and which layer nearly rejected it is
information the ablation needs.

Survivorship here is honest by construction: the set is derived from verdicts already written to
disk by an auditor frozen before the corpus was generated. Nothing is selected on performance beyond
the requirement that the strategy trade at all.

Usage:
    python scripts/tag_survivors.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import get_logger  # noqa: E402

_log = get_logger(__name__)

OUT_DIR = Path("benchmarks/alphaaudit/survivors")

#: The corpora that pool into the Phase 1.4 experiment. Both were generated under the identical
#: prompt digest, which is what makes pooling them one experiment rather than two.
CORPORA: tuple[Path, ...] = (
    Path("runs/20260728T172115Z"),
    Path("runs/batch2"),
)

#: A Sharpe this close to zero means the strategy never took a position. Such candidates executed
#: correctly and are counted in the corpus statistics, but they carry no ordering information and
#: are not strategies anyone could deploy.
FLAT_TOLERANCE = 1e-9


def load_corpus(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Backtest results, audit verdicts and generation metadata for one corpus."""
    backtests_path = run_dir / "backtest_results.json"
    audit_path = run_dir / "audit_results.json"
    if not backtests_path.exists() or not audit_path.exists():
        return [], {}, {}
    backtests = json.loads(backtests_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return backtests, audit, summary


def verdicts_for(name: str, audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each layer's verdict for one candidate, kept apart.

    A layer with no entry for this candidate did not run on it — the statistical layer only sees
    candidates that produced returns. That is recorded as ``ran: False`` rather than as a pass,
    because a layer that never looked has not cleared anything.
    """
    out: dict[str, dict[str, Any]] = {}
    for layer in ("static", "semantic", "statistical"):
        entry = audit.get(layer, {}).get(name)
        if entry is None:
            out[layer] = {"ran": False, "rejected": None}
        else:
            out[layer] = {"ran": True, **entry}
    return out


def is_survivor(verdicts: dict[str, dict[str, Any]]) -> bool:
    """True when no layer that ran rejected this candidate."""
    return not any(v["ran"] and v["rejected"] for v in verdicts.values())


def collect(run_dir: Path) -> list[dict[str, Any]]:
    """Survivor records for one corpus."""
    backtests, audit, summary = load_corpus(run_dir)
    if not backtests:
        _log.warning("%s has no completed backtest and audit; skipped", run_dir)
        return []

    by_index = {c["class_name"]: c for c in summary.get("candidates", [])}
    provenance = {
        "corpus": str(run_dir),
        "model_tag": summary.get("model_tag"),
        "prompt_digest": summary.get("prompt_digest"),
        "base_seed": summary.get("base_seed"),
    }

    survivors: list[dict[str, Any]] = []
    for record in backtests:
        if record["outcome"] != "evaluated" or record["sharpe"] is None:
            continue
        if abs(float(record["sharpe"])) < FLAT_TOLERANCE:
            continue
        verdicts = verdicts_for(record["name"], audit)
        if not is_survivor(verdicts):
            continue
        source = Path(record["path"])
        meta = by_index.get(source.stem, {})
        survivors.append({
            "name": record["name"],
            "source": str(source),
            "sharpe": record["sharpe"],
            "annualised_return": record["annualised_return"],
            "volatility": record["volatility"],
            "max_drawdown": record["max_drawdown"],
            "n_sessions": record["n_sessions"],
            "theme": meta.get("theme"),
            "seed": meta.get("seed"),
            "audit": verdicts,
            "provenance": provenance,
        })
    return survivors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    survivors: list[dict[str, Any]] = []
    for run_dir in CORPORA:
        survivors.extend(collect(run_dir))

    if not survivors:
        _log.error("no survivors found; has the audit run on a completed backtest?")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for record in survivors:
        source = Path(record["source"])
        if source.exists():
            shutil.copy2(source, args.out / f"{record['name']}.py")

    survivors.sort(key=lambda r: float(r["sharpe"]), reverse=True)
    (args.out / "survivors.json").write_text(
        json.dumps({"n": len(survivors), "survivors": survivors}, indent=2), encoding="utf-8"
    )

    _log.info("tagged %d survivors into %s", len(survivors), args.out)
    print(f"survivors: {len(survivors)}")
    print(f"  Sharpe range: {survivors[-1]['sharpe']:.3f} to {survivors[0]['sharpe']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
