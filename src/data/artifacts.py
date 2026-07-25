"""Register of price-series points that are known to be imperfect, and why.

Some events cannot be adjusted away honestly. A capital change whose true ratio cannot be
determined from daily data, a demerger this project does not model, a rights issue, or a genuine
crash that merely *looks* like a split — each leaves a point in the series that is either
unadjusted or unmodellable. Silently leaving them in the panel would be a landmine: a strategy
that trades one of these dates would look like it found an edge, and nothing downstream could tell
that apart from a real one.

So they are labelled instead. An unadjusted-but-registered point is honest; an unadjusted and
invisible one is not. The backtester and the auditor can query this register, and it is the table
that becomes the Limitations section of the write-up.

Reason codes:
    unresolved_split        a declared capital change whose true ratio the prices do not settle
    demerger_unmodeled      value left the company via a spin-off; splits/bonuses only are modelled
    rights_issue_unmodeled  dilutive rights issue, not modelled
    genuine_event_protected a real market event deliberately preserved, not adjusted away
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from src.common.io import read_parquet
from src.common.log import get_logger

_log = get_logger(__name__)

REGISTER_FILENAME = "artifact_register.parquet"

UNRESOLVED_SPLIT = "unresolved_split"
DEMERGER = "demerger_unmodeled"
RIGHTS_ISSUE = "rights_issue_unmodeled"
GENUINE_EVENT = "genuine_event_protected"


@dataclass(frozen=True)
class ArtifactEntry:
    """One flagged point (or window) in the price series."""

    symbol: str
    start_date: date
    end_date: date          # equal to start_date for a single-session flag
    reason: str
    detail: str


# Events identified by hand during Phase 1.0 and confirmed against the price series. Each is
# either a corporate action of a type this project does not model, or a real market event that
# resembles one. Kept explicit rather than inferred, because the distinction between "crash" and
# "capital change" cannot be drawn from a price ratio alone.
CURATED_ENTRIES: tuple[ArtifactEntry, ...] = (
    ArtifactEntry(
        "TATACHEM", date(2020, 3, 4), date(2020, 3, 4), DEMERGER,
        "consumer business demerged into Tata Consumer; price fall is real but reflects value "
        "leaving the company, not a market move",
    ),
    ArtifactEntry(
        "PEL", date(2022, 8, 30), date(2022, 8, 30), DEMERGER,
        "Piramal Pharma demerged; same treatment as TATACHEM",
    ),
    ArtifactEntry(
        "ITC", date(2020, 1, 1), date(2024, 12, 31), DEMERGER,
        "hotels demerger leaves a persistent basis offset against split-adjusted reference series",
    ),
    ArtifactEntry(
        "BHARTIARTL", date(2020, 1, 1), date(2021, 9, 27), RIGHTS_ISSUE,
        "rights issue not modelled; a small persistent offset before the ex-date",
    ),
    ArtifactEntry(
        "RELIANCE", date(2020, 5, 1), date(2020, 6, 30), RIGHTS_ISSUE,
        "2020 rights issue not modelled; dilutive, so a naive return calculation overstates it",
    ),
    ArtifactEntry(
        "YESBANK", date(2020, 3, 6), date(2020, 3, 6), GENUINE_EVENT,
        "RBI moratorium imposed 2020-03-05; the fall is real market history and must not be "
        "adjusted away",
    ),
    ArtifactEntry(
        "IDEA", date(2020, 3, 18), date(2020, 3, 18), GENUINE_EVENT,
        "AGR dues ruling; genuine crash, deliberately preserved",
    ),
)


def build_register(processed_root: Path) -> pl.DataFrame:
    """Assemble the register from pipeline outputs plus the curated entries.

    Unresolved capital changes are read from the snap table rather than hard-coded, so the register
    stays correct if the snapping tolerance or the declared-action feed ever changes.
    """
    entries = list(CURATED_ENTRIES)

    snap_path = processed_root / "snap_table.parquet"
    if snap_path.exists():
        snaps = read_parquet(snap_path).filter(pl.col("outcome") == "unresolved")
        for row in snaps.iter_rows(named=True):
            entries.append(
                ArtifactEntry(
                    row["symbol"], row["ex_date"], row["ex_date"], UNRESOLVED_SPLIT,
                    f"declared factor {row['declared_factor']:.2f} but prices imply "
                    f"{row['implied_factor']:.2f}; the two cannot be separated from daily data, "
                    "so the as-traded price is left unadjusted",
                )
            )

    frame = pl.DataFrame(
        {
            "symbol": [e.symbol for e in entries],
            "start_date": [e.start_date for e in entries],
            "end_date": [e.end_date for e in entries],
            "reason": [e.reason for e in entries],
            "detail": [e.detail for e in entries],
        }
    ).sort(["symbol", "start_date"])
    _log.info("artifact register: %d entries across %d symbols",
              frame.height, frame["symbol"].n_unique())
    return frame


def load_register(processed_root: Path) -> pl.DataFrame:
    """Read the register. Raises if absent — callers must not silently proceed without it."""
    return read_parquet(processed_root / REGISTER_FILENAME)


def is_flagged(register: pl.DataFrame, symbol: str, day: date) -> bool:
    """True if this symbol-date falls inside any registered artifact window.

    The backtester uses this to mark trades that touch known-imperfect data, so an apparent edge
    at such a point can be recognised rather than trusted.
    """
    hit = register.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("start_date") <= day)
        & (pl.col("end_date") >= day)
    )
    return hit.height > 0


def flagged_reasons(register: pl.DataFrame, symbol: str, day: date) -> list[str]:
    """Every reason code covering this symbol-date; empty when the point is clean."""
    hit = register.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("start_date") <= day)
        & (pl.col("end_date") >= day)
    )
    return hit["reason"].to_list()
