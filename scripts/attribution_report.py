"""Report which leak classes the static auditor attributes to a directory of labelled cases.

Recall answers "was this caught". This answers "was it caught for the right reason", which is the
question RQ1 depends on: the paper reports how failures distribute across leakage classes, and a
distribution built from mislabelled findings is wrong even where every accept/reject decision is
right.

Each case declares its true class through its filename prefix, so the expected label is fixed by
the corpus rather than by anything the auditor produces.

**On reading the output.** Run against ``tests/fixtures/refine`` the numbers are training accuracy
on a burned set the auditor was tuned against, and are not a result. Only the locked set, scored
once, gives a figure that means anything.

    python scripts/attribution_report.py tests/fixtures/refine
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.audit.static import Severity, audit_file, is_rejected  # noqa: E402

# Filename prefix -> the leak class the case is an instance of. `future_dependent_ordering` maps to
# `full_sample_statistic` because that is what it mechanically is: a key computed over the whole
# sample driving a selection. It was not given its own class, since a class per surface form would
# make the corpus breakdown a description of how cases were written rather than of how they leak.
CATEGORY_TO_CLASS = {
    "future_indexing": "future_indexing",
    "survivorship_selection": "survivorship_selection",
    "full_sample_fit": "full_sample_fit",
    "full_sample_statistic": "full_sample_statistic",
    "boundary_crossing_window": "boundary_crossing_window",
    "target_in_features": "target_in_features",
    "point_in_time_bypass": "point_in_time_bypass",
    "future_dependent_ordering": "full_sample_statistic",
    "snooped_parameter": "snooped_parameter",
}


def expected_class(stem: str) -> str:
    """The class a case is an instance of, from its filename prefix."""
    for prefix in sorted(CATEGORY_TO_CLASS, key=len, reverse=True):
        if stem.startswith(prefix):
            return CATEGORY_TO_CLASS[prefix]
    raise SystemExit(f"{stem}: filename does not begin with a known category prefix")


def sources(directory: Path) -> list[Path]:
    """Every case file in ``directory``, excluding package plumbing."""
    return [p for p in sorted(directory.glob("*.py")) if not p.name.startswith("__")]


def main(root: Path) -> None:
    leaky, honest = sources(root / "leaky"), sources(root / "honest")
    if not leaky:
        raise SystemExit(f"no cases found under {root / 'leaky'}")

    caught = attributed = 0
    width = max(len(p.stem) for p in leaky)
    for path in leaky:
        want = expected_class(path.stem)
        findings = audit_file(path)
        got = sorted({f.leak_class.value for f in findings if f.severity is Severity.HIGH})
        rejected = is_rejected(findings)
        caught += rejected
        attributed += want in got
        verdict = "ok" if want in got else ("MISSED" if not rejected else "mislabelled")
        print(f"{path.stem:<{width}}  want={want:<24} got={','.join(got) or '-':<62}{verdict}")

    flagged = [p.stem for p in honest if is_rejected(audit_file(p))]
    print(
        f"\ncaught {caught}/{len(leaky)}"
        f"   true class present {attributed}/{len(leaky)}"
        f"   false positives {len(flagged)}/{len(honest)}"
    )
    if flagged:
        print("falsely flagged: " + ", ".join(flagged))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/attribution_report.py <fixture-directory>")
    main(Path(sys.argv[1]))
