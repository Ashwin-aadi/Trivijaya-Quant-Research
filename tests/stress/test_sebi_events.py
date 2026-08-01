"""Tests for the hand-curated SEBI / market-structure event timeline.

The timeline is typed by a human, so it is tested against the mistakes humans make when typing —
a missing citation, a date transposed so a rule takes effect before it was announced, a copied row
left unedited. The shipped file is checked as a fixture in its own right, because it is a research
artefact and not just an input.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.common.exceptions import DataIntegrityError
from src.stress.events import VALID_EVIDENCE, load_events, validate_events


def _minimal(**overrides: object) -> pl.DataFrame:
    row = {
        "announced_date": date(2020, 1, 1),
        "effective_date": date(2020, 2, 1),
        "effective_precision": "day",
        "category": "margin",
        "title": "A rule",
        "description": "What it did.",
        "source_url": "https://example.org/circular.pdf",
        "source_ref": "SEBI/X/1",
        "evidence": "primary_circular_pdf",
    }
    row.update(overrides)
    return pl.DataFrame([row])


# --- the shipped timeline itself -------------------------------------------------


def test_the_shipped_timeline_validates() -> None:
    frame = load_events()
    assert frame.height > 0
    validate_events(frame)


def test_every_shipped_event_is_citable() -> None:
    """No event may rest on nothing. This is the charter's 'never fabricate' rule as a test."""
    frame = load_events()
    assert frame["source_url"].str.starts_with("https://").all()
    assert (frame["source_ref"].str.strip_chars().str.len_chars() > 0).all()
    assert set(frame["evidence"].to_list()) <= VALID_EVIDENCE


def test_shipped_timeline_covers_the_development_window() -> None:
    """Sanity: a timeline with nothing in the study window would be useless to Phase 2.2."""
    frame = load_events()
    in_window = frame.filter(
        (pl.col("effective_date") >= date(2020, 1, 1))
        & (pl.col("effective_date") <= date(2024, 12, 31))
    )
    assert in_window.height >= 5, "too few events inside the development window to be useful"


def test_shipped_timeline_is_sorted_and_unique() -> None:
    frame = load_events()
    assert frame["effective_date"].to_list() == sorted(frame["effective_date"].to_list())
    assert frame["title"].n_unique() == frame.height


# --- the validator catches what hand-editing breaks -------------------------------


def test_rejects_an_event_effective_before_it_was_announced() -> None:
    frame = _minimal(announced_date=date(2020, 6, 1), effective_date=date(2020, 1, 1))
    with pytest.raises(DataIntegrityError, match="before they were announced"):
        validate_events(frame)


def test_rejects_a_missing_citation() -> None:
    with pytest.raises(DataIntegrityError, match="source_url"):
        validate_events(_minimal(source_url="   "))


def test_rejects_a_non_https_source() -> None:
    with pytest.raises(DataIntegrityError, match="https"):
        validate_events(_minimal(source_url="http://example.org/x.pdf"))


def test_rejects_an_unknown_evidence_grade() -> None:
    with pytest.raises(DataIntegrityError, match="evidence"):
        validate_events(_minimal(evidence="trust_me"))


def test_rejects_an_unknown_precision() -> None:
    with pytest.raises(DataIntegrityError, match="precision"):
        validate_events(_minimal(effective_precision="quarter"))


def test_rejects_a_duplicated_row() -> None:
    """A copied row left unedited is the most likely way this file grows a wrong entry."""
    doubled = pl.concat([_minimal(), _minimal()])
    with pytest.raises(DataIntegrityError, match="duplicate"):
        validate_events(doubled)


def test_rejects_an_empty_timeline() -> None:
    with pytest.raises(DataIntegrityError, match="empty"):
        validate_events(_minimal().clear())


def test_rejects_a_missing_column() -> None:
    with pytest.raises(DataIntegrityError, match="missing columns"):
        validate_events(_minimal().drop("source_ref"))
