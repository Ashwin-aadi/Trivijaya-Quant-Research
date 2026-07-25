"""Tests for resolving declared-but-contradicted corporate actions.

The behaviour under test is narrow by design: a price move may settle the *size* of an event the
data feed already reported, but must never be allowed to invent an event. The test that matters
most here is ``test_unexplained_move_is_left_alone`` — if that ever fails, genuine crashes are
being silently flattened into non-events, which is fabricated price history.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.data.calendar import TradingCalendar
from src.data.corporate_actions import (
    FactorDisagreement,
    find_unexplained_moves,
    resolve_disputed,
    snap_to_simple_factor,
)


def weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def test_snaps_combined_bonus_and_split() -> None:
    # A 1:1 bonus alongside a 1:5 split moves the price by 10x, but a feed reporting only the
    # split leg declares 5x. The observed ratio resolves it.
    assert snap_to_simple_factor(9.602) == pytest.approx(10.0)


def test_snaps_plain_ratios() -> None:
    assert snap_to_simple_factor(1.98) == pytest.approx(2.0)
    assert snap_to_simple_factor(4.9) == pytest.approx(5.0)
    assert snap_to_simple_factor(1.02) == pytest.approx(1.0)


def test_refuses_when_nothing_fits() -> None:
    # 4.33 sits too far from both 4.0 and 5.0 to claim either; it must not be forced.
    assert snap_to_simple_factor(4.33) is None
    # 1.10 falls between the 1:5 bonus (1.2) and no-change (1.0), matching neither.
    assert snap_to_simple_factor(1.10) is None


def test_a_crash_is_indistinguishable_from_a_bonus_by_ratio_alone() -> None:
    """Why the declared-action guardrail is load-bearing, stated as a test.

    A stock that falls 40% on bad news produces a ratio of 1.667 — which is exactly the ratio a
    2:3 bonus produces. Nothing in the number itself separates the two. Snapping therefore cannot
    be allowed to run on price moves alone; it is only ever applied where the feed has already
    declared that a corporate action occurred on that date.
    """
    assert snap_to_simple_factor(1.667) == pytest.approx(5 / 3)


def test_resolve_classifies_three_outcomes() -> None:
    disputed = [
        FactorDisagreement("COMBINED", date(2022, 9, 13), 5.0, 9.602),   # -> snapped to 10
        FactorDisagreement("WRONGDATE", date(2023, 9, 29), 2.0, 0.98),   # -> no adjustment
        FactorDisagreement("ODDBALL", date(2020, 4, 3), 5.0, 4.33),      # -> unresolved
    ]
    events, decisions = resolve_disputed(disputed)

    by_symbol = {d.symbol: d for d in decisions}
    assert by_symbol["COMBINED"].outcome == "snapped"
    assert by_symbol["COMBINED"].applied_factor == pytest.approx(10.0)
    assert by_symbol["WRONGDATE"].outcome == "no_adjustment"
    assert by_symbol["WRONGDATE"].applied_factor is None
    assert by_symbol["ODDBALL"].outcome == "unresolved"
    assert by_symbol["ODDBALL"].applied_factor is None

    # Only the snapped case produces an adjustment; the other two change nothing.
    assert [e.symbol for e in events] == ["COMBINED"]

    # Every input is accounted for in the table the reviewer reads.
    assert len(decisions) == len(disputed)


def test_unexplained_move_is_left_alone() -> None:
    """A large fall with no declared action must be surfaced, never corrected.

    This is the guardrail. A genuine crash on bad news looks exactly like a 1:2 split in the price
    series; the only thing distinguishing them is whether a corporate action was declared. Treating
    an undeclared move as a split would erase a real event from history.
    """
    sessions = weekdays(date(2024, 1, 1), 4)
    calendar = TradingCalendar(sessions)
    panel = pl.DataFrame(
        {
            "symbol": ["CRASH"] * 4,
            "session_date": sessions,
            # Halves on the third session with no corporate action behind it.
            "adj_close": [100.0, 100.0, 50.0, 50.0],
        }
    )
    found = find_unexplained_moves(panel, calendar, declared_dates=set())
    assert found.height == 1
    assert found["symbol"][0] == "CRASH"
    assert found["session_date"][0] == sessions[2]

    # Same move, but the feed declared an action that day: already handled, so not re-surfaced.
    handled = find_unexplained_moves(
        panel, calendar, declared_dates={("CRASH", sessions[2])}
    )
    assert handled.height == 0


def test_non_adjacent_sessions_are_not_flagged() -> None:
    # A symbol that stops trading and resumes later must not look like a capital change.
    sessions = weekdays(date(2024, 1, 1), 6)
    calendar = TradingCalendar(sessions)
    panel = pl.DataFrame(
        {
            "symbol": ["GAPPY"] * 2,
            "session_date": [sessions[0], sessions[4]],
            "adj_close": [100.0, 40.0],
        }
    )
    assert find_unexplained_moves(panel, calendar, declared_dates=set()).height == 0
