"""Tests for the trading-calendar engine.

These use a small, hand-built session set so the arithmetic is checkable by eye — no network,
no dependence on a real holiday source. They cover the two operations the backtester relies on
(shift for next-open execution, require_trading_day for closed-day protection) plus the boundary
behaviour that tends to hide off-by-one bugs.
"""

from datetime import date

import pytest

from src.data.calendar import CalendarError, TradingCalendar, sessions_from_weekdays


def make_week_calendar() -> TradingCalendar:
    # Mon 2024-01-01 through Fri 2024-01-05, then Mon 2024-01-08 (skipping the weekend).
    days = [date(2024, 1, d) for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)]
    return TradingCalendar(days, name="TEST")


def test_membership_and_weekend_gap() -> None:
    cal = make_week_calendar()
    assert cal.is_trading_day(date(2024, 1, 5))
    assert not cal.is_trading_day(date(2024, 1, 6))   # Saturday
    assert not cal.is_trading_day(date(2024, 1, 7))   # Sunday


def test_next_and_previous_cross_weekend() -> None:
    cal = make_week_calendar()
    # Friday's next session is the following Monday, not Saturday.
    assert cal.next_session(date(2024, 1, 5)) == date(2024, 1, 8)
    assert cal.previous_session(date(2024, 1, 8)) == date(2024, 1, 5)


def test_next_session_on_or_after() -> None:
    cal = make_week_calendar()
    # A Saturday resolves forward to Monday whether or not we allow "on".
    assert cal.next_session(date(2024, 1, 6), strictly_after=False) == date(2024, 1, 8)
    # On a trading day, strictly_after=False returns the day itself.
    assert cal.next_session(date(2024, 1, 5), strictly_after=False) == date(2024, 1, 5)


def test_shift_is_next_open_execution() -> None:
    cal = make_week_calendar()
    # Signal at Friday's close -> fill at the next session's open (Monday).
    assert cal.shift(date(2024, 1, 5), 1) == date(2024, 1, 8)
    # Five sessions back from the second Monday lands on the first Monday.
    assert cal.shift(date(2024, 1, 8), -5) == date(2024, 1, 1)


def test_shift_requires_trading_day() -> None:
    cal = make_week_calendar()
    with pytest.raises(CalendarError):
        cal.shift(date(2024, 1, 6), 1)  # shifting from a Saturday is a bug, not a no-op


def test_shift_out_of_range_raises() -> None:
    cal = make_week_calendar()
    with pytest.raises(CalendarError):
        cal.shift(date(2024, 1, 12), 1)  # nothing after the last session


def test_require_trading_day_raises_on_holiday() -> None:
    cal = make_week_calendar()
    with pytest.raises(CalendarError):
        cal.require_trading_day(date(2024, 1, 6))


def test_sessions_in_range_inclusive() -> None:
    cal = make_week_calendar()
    got = cal.sessions_in_range(date(2024, 1, 4), date(2024, 1, 9))
    assert got == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]
    assert cal.count_sessions(date(2024, 1, 4), date(2024, 1, 9)) == 4


def test_empty_calendar_rejected() -> None:
    with pytest.raises(CalendarError):
        TradingCalendar([])


def test_out_of_range_query_rejected() -> None:
    cal = make_week_calendar()
    with pytest.raises(CalendarError):
        cal.next_session(date(2023, 12, 1))


def test_sessions_from_weekdays_removes_weekends_and_holidays() -> None:
    # Build Mon-Fri for the first full week, declaring Wednesday a holiday.
    got = sessions_from_weekdays(
        start=date(2024, 1, 1),
        end=date(2024, 1, 7),
        holidays=[date(2024, 1, 3)],
    )
    assert got == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)]
