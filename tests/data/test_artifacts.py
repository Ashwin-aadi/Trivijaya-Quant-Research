"""Tests for the known-artifacts register.

The register exists so a strategy that trades a known-imperfect point can be recognised rather
than trusted. These tests check the query surface the backtester relies on, and that unresolved
capital changes are picked up from the pipeline rather than hand-maintained.
"""

from datetime import date
from pathlib import Path

import polars as pl

from src.data.artifacts import (
    GENUINE_EVENT,
    UNRESOLVED_SPLIT,
    build_register,
    flagged_reasons,
    is_flagged,
)


def test_curated_entries_are_present(tmp_path: Path) -> None:
    register = build_register(tmp_path)          # no snap table -> curated entries only
    symbols = set(register["symbol"].to_list())
    assert {"YESBANK", "IDEA", "TATACHEM", "PEL", "ITC", "RELIANCE", "BHARTIARTL"} <= symbols


def test_unresolved_splits_come_from_the_pipeline(tmp_path: Path) -> None:
    # Written by build_corporate_actions.py; the register must pick it up rather than hard-code it,
    # so changing the snapping tolerance cannot leave the register stale.
    pl.DataFrame(
        {
            "symbol": ["IRCON", "COCHINSHIP"],
            "ex_date": [date(2020, 4, 3), date(2024, 1, 10)],
            "declared_factor": [5.0, 2.0],
            "implied_factor": [4.33, 1.667],
            "applied_factor": [None, 1.667],
            "outcome": ["unresolved", "snapped"],
        }
    ).write_parquet(tmp_path / "snap_table.parquet")

    register = build_register(tmp_path)
    unresolved = register.filter(pl.col("reason") == UNRESOLVED_SPLIT)
    # Only the unresolved event is registered; a successfully snapped one is not an artifact.
    assert unresolved["symbol"].to_list() == ["IRCON"]
    assert "4.33" in unresolved["detail"][0]


def test_point_queries(tmp_path: Path) -> None:
    register = build_register(tmp_path)
    assert is_flagged(register, "YESBANK", date(2020, 3, 6))
    assert not is_flagged(register, "YESBANK", date(2020, 3, 9))
    assert not is_flagged(register, "INFY", date(2020, 3, 6))
    assert flagged_reasons(register, "YESBANK", date(2020, 3, 6)) == [GENUINE_EVENT]
    assert flagged_reasons(register, "INFY", date(2020, 3, 6)) == []


def test_window_entries_cover_their_whole_range(tmp_path: Path) -> None:
    # RELIANCE's rights issue is registered as a window, not a single session.
    register = build_register(tmp_path)
    assert is_flagged(register, "RELIANCE", date(2020, 5, 1))
    assert is_flagged(register, "RELIANCE", date(2020, 6, 15))
    assert is_flagged(register, "RELIANCE", date(2020, 6, 30))
    assert not is_flagged(register, "RELIANCE", date(2020, 7, 1))
