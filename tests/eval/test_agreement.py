"""Tests for Cohen's kappa and for the hand-labelling sheet it will be computed over.

Two halves. The first pins the statistic against cases whose answer is known without running the
code — perfect agreement, chance-level agreement, systematic opposition, and one 2x2 table worked
out by hand in a comment. A kappa implementation that is subtly wrong would still return plausible
numbers, so "it ran" is not evidence of anything and every assertion here has an arithmetic reason
behind it.

The second half checks the properties of the sheet that make the measurement valid at all: exactly
fifty items, ids that do not collide, byte-identical output on a rerun, and — the one that matters
most — no trace anywhere in the sheet of what any item is expected to be.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import pytest
from scripts.build_label_sheet import (
    SHEET_SIZE,
    build_items,
    render_markdown,
    write_csv,
)

from src.audit.prompts import LABELS
from src.eval.agreement import (
    UNDEFINED_KAPPA,
    agreement_summary,
    cohens_kappa,
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

TWO = ("yes", "no")


def _repeat(pairs: list[tuple[str, str, int]]) -> tuple[list[str], list[str]]:
    """Expand [(label_a, label_b, count), ...] into two aligned label sequences."""
    a: list[str] = []
    b: list[str] = []
    for label_a, label_b, count in pairs:
        a += [label_a] * count
        b += [label_b] * count
    return a, b


# --- the statistic -------------------------------------------------------------


def test_perfect_agreement_gives_kappa_one() -> None:
    labels = ["consistent", "unfalsifiable_mechanism", "consistent", "consistent"]
    assert cohens_kappa(labels, list(labels), LABELS) == pytest.approx(1.0)


def test_chance_level_agreement_gives_zero() -> None:
    # Each rater uses both categories half the time and the four cells are equal, so observed
    # agreement (0.5) is exactly what independence predicts (0.5*0.5 + 0.5*0.5 = 0.5).
    a, b = _repeat([("yes", "yes", 25), ("yes", "no", 25), ("no", "yes", 25), ("no", "no", 25)])
    assert cohens_kappa(a, b, TWO) == pytest.approx(0.0, abs=1e-12)


def test_systematic_disagreement_gives_a_negative_kappa() -> None:
    # The raters never agree, yet their marginals predict 50% agreement by chance. Kappa is
    # therefore (0 - 0.5) / (1 - 0.5) = -1.0: opposition, not noise.
    a, b = _repeat([("yes", "no", 10), ("no", "yes", 10)])
    assert cohens_kappa(a, b, TWO) == pytest.approx(-1.0)


def test_known_answer_two_by_two() -> None:
    # Hand computation, n = 50:
    #   cells      a=yes,b=yes 20 | a=yes,b=no  5
    #              a=no ,b=yes 10 | a=no ,b=no 15
    #   p_o = (20 + 15) / 50                      = 0.70
    #   rater A marginals: yes 25, no 25          -> 0.50, 0.50
    #   rater B marginals: yes 30, no 20          -> 0.60, 0.40
    #   p_e = 0.50*0.60 + 0.50*0.40               = 0.50
    #   kappa = (0.70 - 0.50) / (1 - 0.50)        = 0.40
    a, b = _repeat([("yes", "yes", 20), ("yes", "no", 5), ("no", "yes", 10), ("no", "no", 15)])
    assert cohens_kappa(a, b, TWO) == pytest.approx(0.40)


def test_confusion_matrix_records_each_pair_in_category_order() -> None:
    a, b = _repeat([("yes", "yes", 20), ("yes", "no", 5), ("no", "yes", 10), ("no", "no", 15)])
    assert confusion_matrix(a, b, TWO) == [[20, 5], [10, 15]]


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="different numbers of items"):
        cohens_kappa(["yes", "no"], ["yes"], TWO)


def test_unknown_category_raises() -> None:
    # A label outside the taxonomy is a defect in whatever produced it. Silently folding it into a
    # nearest category would move the very number this module exists to measure.
    with pytest.raises(ValueError, match="outside the agreed categories"):
        cohens_kappa(["yes", "maybe"], ["yes", "no"], TWO)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="zero items"):
        cohens_kappa([], [], TWO)


def test_duplicate_categories_raise() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        cohens_kappa(["yes"], ["yes"], ("yes", "yes", "no"))


def test_single_shared_category_is_undefined_not_a_crash() -> None:
    # Both raters answer `consistent` to everything. Observed agreement is 1 and expected agreement
    # is also 1, so kappa is 0/0. This is the realistic failure mode for a small local model, and
    # it must report as undefined rather than raising ZeroDivisionError or being read as 1.0.
    labels = ["consistent"] * 10
    result = cohens_kappa(labels, list(labels), LABELS)
    assert math.isnan(result)
    assert math.isnan(UNDEFINED_KAPPA)


def test_summary_reports_kappa_raw_agreement_counts_and_n() -> None:
    a, b = _repeat([("yes", "yes", 20), ("yes", "no", 5), ("no", "yes", 10), ("no", "no", 15)])
    summary = agreement_summary(a, b, TWO)
    assert summary.n == 50
    assert summary.raw_agreement == pytest.approx(0.70)
    assert summary.kappa == pytest.approx(0.40)
    by_name = {c.category: c for c in summary.per_category}
    assert (by_name["yes"].count_a, by_name["yes"].count_b, by_name["yes"].agreed) == (25, 30, 20)
    assert (by_name["no"].count_a, by_name["no"].count_b, by_name["no"].agreed) == (25, 20, 15)


def test_summary_kappa_matches_the_standalone_function() -> None:
    # Two entry points, one implementation: they must never report different numbers for the same
    # data, because a checkpoint report will quote whichever was convenient.
    a, b = _repeat([("yes", "yes", 3), ("yes", "no", 7), ("no", "yes", 2), ("no", "no", 8)])
    assert agreement_summary(a, b, TWO).kappa == pytest.approx(cohens_kappa(a, b, TWO))


def test_summary_counts_every_category_including_unused_ones() -> None:
    # An unused class is a finding in itself — it means the sample could not test that label — so
    # it has to appear with a zero rather than be dropped from the table.
    labels = ["consistent"] * 4 + ["unfalsifiable_mechanism"] * 4
    summary = agreement_summary(labels, list(labels), LABELS)
    assert len(summary.per_category) == len(LABELS)
    unused = {c.category for c in summary.per_category if c.count_a == 0 and c.count_b == 0}
    assert "rationale_implementation_mismatch" in unused


# --- the sheet -----------------------------------------------------------------


def test_sheet_has_exactly_fifty_items_with_unique_ids() -> None:
    items = build_items(REPO_ROOT)
    assert len(items) == SHEET_SIZE
    assert len({item.item_id for item in items}) == SHEET_SIZE


def test_sheet_is_reproducible_across_two_runs_with_the_same_seed() -> None:
    first = build_items(REPO_ROOT, seed=42)
    second = build_items(REPO_ROOT, seed=42)
    assert [i.item_id for i in first] == [i.item_id for i in second]
    assert [i.rationale for i in first] == [i.rationale for i in second]
    assert [i.code_excerpt for i in first] == [i.code_excerpt for i in second]


def test_a_different_seed_reorders_the_same_items() -> None:
    # Guards against a shuffle that silently does nothing: the set must be identical, the order
    # must not be, or the seeded shuffle is not doing the de-anchoring job it is there for.
    baseline = build_items(REPO_ROOT, seed=42)
    other = build_items(REPO_ROOT, seed=7)
    assert {i.item_id for i in baseline} == {i.item_id for i in other}
    assert [i.item_id for i in baseline] != [i.item_id for i in other]


def test_sheet_draws_thirty_clean_three_leaky_and_seventeen_substituted() -> None:
    items = build_items(REPO_ROOT)
    counts = Counter(item.origin for item in items)
    assert counts == {"clean": 30, "leaky": 3, "constructed": 17}


def test_no_item_reveals_an_expected_label() -> None:
    # The whole point of the sheet. A reviewer who has been shown the answer is not an independent
    # rater, and a kappa computed against an anchored reviewer measures nothing at all.
    items = build_items(REPO_ROOT)
    for item in items:
        text = f"{item.rationale}\n{item.code_excerpt}"
        for label in LABELS:
            assert not re.search(rf"\b{re.escape(label)}\b", text), (
                f"item {item.item_id} names the label {label!r}"
            )


def test_no_excerpt_carries_the_fixtures_own_rationale() -> None:
    # Leaving the class-level `rationale` attribute in the code would print the original rationale
    # directly under the substituted one, giving away every constructed item at a glance.
    for item in build_items(REPO_ROOT):
        assert "rationale" not in item.code_excerpt


def test_trimmed_excerpts_say_that_they_were_trimmed() -> None:
    for item in build_items(REPO_ROOT):
        assert item.trimmed == ("# [excerpt:" in item.code_excerpt)


def test_markdown_shows_every_item_with_a_blank_label_line() -> None:
    items = build_items(REPO_ROOT)
    markdown = render_markdown(items)
    # The underscores distinguish the fifty answer lines from the header's reference to them.
    assert markdown.count("Your label: ___") == SHEET_SIZE
    for item in items:
        assert item.item_id in markdown
    header, body = markdown.split("\n### Item ", 1)
    for label in LABELS:
        # Defined once in the header legend, and named nowhere among the fifty items.
        assert re.search(rf"\b{re.escape(label)}\b", header)
        assert not re.search(rf"\b{re.escape(label)}\b", body)


def test_csv_has_the_expected_columns_and_an_empty_label_column(tmp_path: Path) -> None:
    items = build_items(REPO_ROOT)
    destination = tmp_path / "label_sheet.csv"
    write_csv(items, destination)
    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == SHEET_SIZE
    assert list(rows[0]) == ["item_id", "source_file", "rationale", "code_excerpt", "human_label"]
    assert all(row["human_label"] == "" for row in rows)
    assert all(row["rationale"].strip() for row in rows)
    assert all(row["code_excerpt"].strip() for row in rows)
