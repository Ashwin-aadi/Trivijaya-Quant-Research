"""Classify the hand-labelled items with the semantic auditor and measure agreement.

Two modes, and the estimate mode exists because the charter requires a wall-clock projection before
any large batch of local inference is started:

    python scripts/run_semantic_kappa.py --estimate   # time a few calls, project the full batch
    python scripts/run_semantic_kappa.py              # classify all items and compute kappa

The reviewer's labels are read from ``benchmarks/alphaaudit/label_sheet.csv``. Their provenance is
recorded in DECISIONS.md as reviewer-confirmed over a drafted first pass, which is a weaker
reference than labels produced from scratch and is stated wherever kappa is quoted.

No prompt is tuned here, and none may be tuned after kappa is known. A poor agreement score is a
reported finding about small local models, not a signal to adjust the prompt until the number
improves — that would be selecting the conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit.prompts import LABELS  # noqa: E402
from src.audit.semantic import (  # noqa: E402
    MODEL_TAG,
    SemanticAuditParseError,
    SemanticAuditUnavailable,
    classify,
    is_available,
)
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.eval.agreement import agreement_summary  # noqa: E402

_log = get_logger("semantic_kappa")

SHEET = Path("benchmarks/alphaaudit/label_sheet.csv")
# Enough calls to see the spread without spending real time on a probe. The charter asks for ten.
ESTIMATE_CALLS = 10


def load_items() -> list[dict[str, str]]:
    """Read the sheet, keeping only rows the reviewer actually labelled."""
    rows = list(csv.DictReader(SHEET.read_text(encoding="utf-8").splitlines()))
    labelled = [r for r in rows if r.get("human_label", "").strip()]
    if not labelled:
        raise SystemExit(f"no human labels found in {SHEET}; fill the human_label column first")
    unknown = {r["human_label"] for r in labelled} - set(LABELS)
    if unknown:
        raise SystemExit(f"sheet contains labels outside the taxonomy: {sorted(unknown)}")
    return labelled


def estimate(items: list[dict[str, str]]) -> None:
    """Time a handful of real calls and project the full batch, without running it."""
    sample = items[:ESTIMATE_CALLS]
    durations: list[float] = []
    for index, row in enumerate(sample, start=1):
        started = time.perf_counter()
        classify(row["rationale"], row["code_excerpt"])
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        _log.info("call %d/%d took %.1fs", index, len(sample), elapsed)

    mean = sum(durations) / len(durations)
    slowest = max(durations)
    projected = mean * len(items)
    print(f"\nmodel: {MODEL_TAG}")
    print(f"timed calls:        {len(durations)}")
    print(f"mean per item:      {mean:.1f}s")
    print(f"slowest observed:   {slowest:.1f}s")
    print(f"items to classify:  {len(items)}")
    print(f"PROJECTED WALL CLOCK: {projected / 60:.1f} min "
          f"(worst case {slowest * len(items) / 60:.1f} min)")
    print("\nThe charter requires this figure to be reported before a batch exceeding "
          "30 minutes is started.")


def run(items: list[dict[str, str]]) -> None:
    """Classify every item and report agreement against the reviewer's labels."""
    cfg = load_config()
    human: list[str] = []
    model: list[str] = []
    failures: list[tuple[str, str]] = []

    with RunManifest(cfg, script="scripts/run_semantic_kappa.py") as manifest:
        manifest.add_model(MODEL_TAG)
        started = time.perf_counter()
        for index, row in enumerate(items, start=1):
            try:
                finding = classify(row["rationale"], row["code_excerpt"])
            except (SemanticAuditUnavailable, SemanticAuditParseError) as exc:
                # Recorded, never silently mapped to a label. A failure to classify is not a
                # classification of "consistent", and counting it as one would flatter agreement.
                failures.append((row["item_id"], f"{type(exc).__name__}: {exc}"))
                _log.warning("item %s failed: %s", row["item_id"], exc)
                continue
            human.append(row["human_label"])
            model.append(finding.label)
            if index % 10 == 0:
                _log.info("%d/%d classified", index, len(items))

        summary = agreement_summary(human, model, list(LABELS))
        manifest.note("items_classified", len(model))
        manifest.note("items_failed", len(failures))
        manifest.note("cohens_kappa", summary.kappa)
        manifest.note("raw_agreement", summary.raw_agreement)
        elapsed = time.perf_counter() - started

    out = {
        "model_tag": MODEL_TAG,
        "items_classified": len(model),
        "items_failed": len(failures),
        "failures": failures,
        "cohens_kappa": summary.kappa,
        "raw_agreement": summary.raw_agreement,
        "wall_clock_seconds": elapsed,
        "human_labels": human,
        "model_labels": model,
    }
    Path("data/processed/semantic_kappa.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print(f"\nmodel: {MODEL_TAG}")
    print(f"classified: {len(model)}   failed: {len(failures)}")
    print(f"raw agreement: {summary.raw_agreement:.4f}")
    print(f"COHEN'S KAPPA: {summary.kappa:.4f}")
    print(f"wall clock: {elapsed / 60:.1f} min")
    if failures:
        print(f"\nfailures (never counted as a label): {failures[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", action="store_true",
                        help="time a few calls and project the batch, without running it")
    args = parser.parse_args()

    if not is_available():
        raise SystemExit(
            "Ollama is not reachable. This is a halt, not something to work around: no other "
            "model may be substituted, because agreement measured against a different model "
            "answers a different question."
        )
    items = load_items()
    _log.info("%d labelled items in %s", len(items), SHEET)
    if args.estimate:
        estimate(items)
    else:
        run(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
