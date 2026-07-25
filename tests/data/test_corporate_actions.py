"""Tests for split/bonus detection and price adjustment.

The panels here are hand-built so each expected factor is obvious. The most important test is
``test_gap_in_panel_is_rejected``: a hole in the data makes an ordinary price move look like a
stock split, and that error would silently corrupt every adjusted series downstream.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.common.exceptions import DataIntegrityError
from src.data.calendar import TradingCalendar
from src.data.corporate_actions import (
    adjustment_factors,
    apply_adjustments,
    assert_panel_contiguous,
    detect_adjustments,
)


def weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def build_panel(sessions: list[date], closes: list[float], prev_closes: list[float],
                symbol: str = "TESTCO") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_date": sessions,
            "symbol": [symbol] * len(sessions),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "prev_close": prev_closes,
            "volume": [1000.0] * len(sessions),
            "turnover_inr": [c * 1000 for c in closes],
        }
    )


def test_detects_one_for_one_bonus() -> None:
    sessions = weekdays(date(2024, 1, 1), 5)
    cal = TradingCalendar(sessions)
    # Price halves on session index 3: a 1:1 bonus. NSE restates prev_close to the new basis,
    # so prev_close (100) is half the prior session's close (200) -> factor 2.0.
    closes = [200.0, 200.0, 200.0, 100.0, 100.0]
    prev = [200.0, 200.0, 200.0, 100.0, 100.0]
    events = detect_adjustments(build_panel(sessions, closes, prev), cal)
    assert len(events) == 1
    assert events[0].ex_date == sessions[3]
    assert events[0].factor == pytest.approx(2.0)


def test_ordinary_price_move_is_not_an_event() -> None:
    sessions = weekdays(date(2024, 1, 1), 5)
    cal = TradingCalendar(sessions)
    # A 3% drop is a normal day; prev_close tracks the prior close exactly.
    closes = [100.0, 97.0, 99.0, 96.0, 98.0]
    prev = [100.0, 100.0, 97.0, 99.0, 96.0]
    assert detect_adjustments(build_panel(sessions, closes, prev), cal) == []


def test_gap_in_panel_is_rejected() -> None:
    """A missing session must raise, not silently produce a fictitious split."""
    sessions = weekdays(date(2024, 1, 1), 6)
    cal = TradingCalendar(sessions)
    kept = [s for i, s in enumerate(sessions) if i != 3]        # punch a hole
    closes = [100.0] * len(kept)
    panel = build_panel(kept, closes, closes)
    with pytest.raises(DataIntegrityError, match="missing"):
        assert_panel_contiguous(panel, cal)
    with pytest.raises(DataIntegrityError):
        detect_adjustments(panel, cal)


def test_adjustment_factors_restate_history_onto_current_basis() -> None:
    sessions = weekdays(date(2024, 1, 1), 5)
    cal = TradingCalendar(sessions)
    closes = [200.0, 200.0, 200.0, 100.0, 100.0]
    events = detect_adjustments(build_panel(sessions, closes, closes), cal)
    factors = adjustment_factors(events, "TESTCO", sessions)
    # Sessions before the ex-date are divided by 2; the ex-date and later are left alone.
    assert factors[sessions[0]] == pytest.approx(2.0)
    assert factors[sessions[2]] == pytest.approx(2.0)
    assert factors[sessions[3]] == pytest.approx(1.0)
    assert factors[sessions[4]] == pytest.approx(1.0)


def test_apply_adjustments_makes_series_continuous() -> None:
    sessions = weekdays(date(2024, 1, 1), 5)
    cal = TradingCalendar(sessions)
    closes = [200.0, 200.0, 200.0, 100.0, 100.0]
    panel = build_panel(sessions, closes, closes)
    events = detect_adjustments(panel, cal)
    adjusted = apply_adjustments(panel, events).sort("session_date")

    series = adjusted["adj_close"].to_list()
    # The cosmetic halving is gone: the adjusted series is flat across the ex-date.
    assert series == pytest.approx([100.0] * 5)

    # Traded value must survive the adjustment untouched. Splitting a share changes the price and
    # the share count in opposite directions; it does not change how much money changed hands, so
    # adj_close * adj_volume has to equal the raw close * volume session by session. This is what
    # keeps the universe's liquidity ranking unaffected by cosmetic capital changes.
    adjusted_value = [p * v for p, v in
                      zip(series, adjusted["adj_volume"].to_list(), strict=True)]
    raw_value = [p * v for p, v in
                 zip(adjusted["close"].to_list(), adjusted["volume"].to_list(), strict=True)]
    assert adjusted_value == pytest.approx(raw_value)


def test_no_events_still_produces_adjusted_columns() -> None:
    sessions = weekdays(date(2024, 1, 1), 3)
    closes = [50.0, 51.0, 52.0]
    adjusted = apply_adjustments(build_panel(sessions, closes, closes), [])
    assert adjusted["adj_close"].to_list() == closes
