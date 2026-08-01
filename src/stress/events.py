"""Load and validate the hand-curated SEBI / market-structure event timeline.

Separate from the publishing script so the validation rules are importable and testable. The
timeline is the one artefact in this project that is typed by a human rather than fetched, so it
has failure modes no fetched file has — a transposed date, a dropped citation, a copy-pasted row
that was never edited. Those are what :func:`validate_events` looks for.

The canonical copy is version-controlled at ``benchmarks/regimestress/sebi_events.csv``;
``scripts/build_sebi_events.py`` validates it and publishes it to ``data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.common.exceptions import DataIntegrityError

#: Version-controlled canonical timeline.
SOURCE_CSV = Path("benchmarks/regimestress/sebi_events.csv")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "announced_date",
    "effective_date",
    "effective_precision",
    "category",
    "title",
    "description",
    "source_url",
    "source_ref",
    "evidence",
)

#: ``month`` means the effective date is accurate to the month only — used where a rule took effect
#: "from the October 2019 expiry" and the exact session was not sourced. Recording the precision
#: stops a month-accurate date being read later as if it were day-accurate.
VALID_PRECISION = frozenset({"day", "month"})

#: How strongly each row is evidenced. Kept as data rather than prose so the mix can be reported:
#: a timeline where everything rests on secondary sources is a different artefact from one anchored
#: in circular PDFs, and a reader is entitled to know which they have.
VALID_EVIDENCE = frozenset(
    {
        "primary_circular_pdf",
        "primary_press_release",
        "circular_number_cited_by_exchange",
        "secondary_sources_agree",
        "verified_in_phase_1_1",
    }
)


def load_events(path: Path = SOURCE_CSV) -> pl.DataFrame:
    """Read the timeline, validate it, and return it sorted by effective date."""
    if not path.exists():
        raise DataIntegrityError(f"{path} not found")
    frame = pl.read_csv(path, try_parse_dates=True).sort("effective_date")
    validate_events(frame)
    return frame


def validate_events(frame: pl.DataFrame) -> None:
    """Raise ``DataIntegrityError`` on anything a hand-edited timeline plausibly gets wrong."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataIntegrityError(f"event timeline is missing columns: {missing}")
    if frame.height == 0:
        raise DataIntegrityError("event timeline is empty")

    for column in ("source_url", "source_ref", "title", "description"):
        blank = frame.filter(pl.col(column).is_null() | (pl.col(column).str.strip_chars() == ""))
        if blank.height:
            raise DataIntegrityError(
                f"{blank.height} event(s) have an empty {column}; every event must be citable"
            )

    bad_precision = set(frame["effective_precision"].to_list()) - VALID_PRECISION
    if bad_precision:
        raise DataIntegrityError(f"unknown effective_precision values: {sorted(bad_precision)}")

    bad_evidence = set(frame["evidence"].to_list()) - VALID_EVIDENCE
    if bad_evidence:
        raise DataIntegrityError(f"unknown evidence values: {sorted(bad_evidence)}")

    # An effective date before its announcement is the classic transcription slip here.
    inverted = frame.filter(pl.col("effective_date") < pl.col("announced_date"))
    if inverted.height:
        raise DataIntegrityError(
            f"{inverted.height} event(s) take effect before they were announced: "
            f"{inverted['title'].to_list()}"
        )

    if not frame["source_url"].str.starts_with("https://").all():
        raise DataIntegrityError("every source_url must be https")

    duplicated = frame.group_by("title").len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise DataIntegrityError(
            f"duplicate event titles, which usually means a copied row was never edited: "
            f"{duplicated['title'].to_list()}"
        )
