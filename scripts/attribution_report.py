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

# Filename prefix -> the leak class the case is an instance of.
CATEGORY_TO_CLASS = {
    "future_indexing": "future_indexing",
    "survivorship_selection": "survivorship_selection",
    "full_sample_fit": "full_sample_fit",
    "full_sample_statistic": "full_sample_statistic",
    "boundary_crossing_window": "boundary_crossing_window",
    "target_in_features": "target_in_features",
    "point_in_time_bypass": "point_in_time_bypass",
    "future_dependent_ordering": "future_dependent_ordering",
    "snooped_parameter": "snooped_parameter",
}

# Categories no detector can attribute, counted as unattributed and labelled so in the output. They
# are not detector failures and they are not excused either — the case is still expected to be
# rejected, and only the reason is missing.
#
# `target_in_features` — identifying which column is the label is semantic. Nothing structural
# separates a float column that happens to be the prediction target from any other. The previous
# rule matched the word `target`, which is what rejected an honest strategy for naming a local
# `target_weight`. PI ruling, 2026-07-28: report as permanently unattributable, do not merge into
# `future_indexing` to tidy the taxonomy.
#
# `future_dependent_ordering` — this is a surface form, not a mechanism. Two of its three variants
# reduce to a full-sample statistic (an extremum or sort key computed over the whole period) and
# the third reduces to a snooped parameter (scoring candidate windows and keeping the winner). It
# was briefly mapped wholesale to `full_sample_statistic`; that collapse was settled by what the
# auditor already emitted rather than by the mechanics, and was reverted. Splitting it per variant
# now, after seeing the output, would repeat the same error.
UNATTRIBUTABLE = frozenset({"target_in_features", "future_dependent_ordering"})


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

    caught = attributed = by_design = 0
    width = max(len(p.stem) for p in leaky)
    for path in leaky:
        want = expected_class(path.stem)
        findings = audit_file(path)
        got = sorted({f.leak_class.value for f in findings if f.severity is Severity.HIGH})
        rejected = is_rejected(findings)
        caught += rejected
        attributed += want in got
        by_design += want in UNATTRIBUTABLE
        if want in got:
            verdict = "ok"
        elif not rejected:
            verdict = "MISSED"
        elif want in UNATTRIBUTABLE:
            verdict = "unattributable by design"
        else:
            verdict = "mislabelled"
        print(f"{path.stem:<{width}}  want={want:<26} got={','.join(got) or '-':<62}{verdict}")

    flagged = [p.stem for p in honest if is_rejected(audit_file(p))]
    print(
        f"\ncaught {caught}/{len(leaky)}"
        f"   true class present {attributed}/{len(leaky)}"
        f"   ({by_design} of the remainder unattributable by design)"
        f"   false positives {len(flagged)}/{len(honest)}"
    )
    if flagged:
        print("falsely flagged: " + ", ".join(flagged))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/attribution_report.py <fixture-directory>")
    main(Path(sys.argv[1]))
