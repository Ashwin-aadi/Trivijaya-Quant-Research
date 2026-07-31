"""Tradability constraints: whether an order could have been executed at all."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.common.config import ConstraintsConfig, load_config
from src.costs.constraints import (
    ConstraintChecker,
    TradabilityError,
    average_daily_value,
    check_participation,
    within_circuit,
)


@pytest.fixture(scope="module")
def limits() -> ConstraintsConfig:
    return load_config().constraints


@pytest.fixture
def panel() -> pl.DataFrame:
    """Thirty sessions of one liquid name and one illiquid one."""
    sessions = [date(2024, 1, 1 + i) for i in range(30)]
    return pl.DataFrame({
        "symbol": ["LIQUID"] * 30 + ["THIN"] * 30,
        "session_date": sessions * 2,
        "close": [100.0] * 30 + [50.0] * 30,
        "volume": [1_000_000.0] * 30 + [100.0] * 30,
    })


def test_within_circuit_accepts_the_band_edge() -> None:
    """A stock locked at its circuit still trades at that price; it cannot trade beyond it."""
    assert within_circuit(120.0, 100.0, 0.20) is True
    assert within_circuit(80.0, 100.0, 0.20) is True
    assert within_circuit(120.01, 100.0, 0.20) is False
    assert within_circuit(79.99, 100.0, 0.20) is False


def test_within_circuit_rejects_a_nonsense_previous_close() -> None:
    assert within_circuit(100.0, 0.0, 0.20) is False


def test_participation_within_the_limit_returns_the_rate() -> None:
    rate = check_participation(500.0, 100_000.0, limit=0.01)
    assert rate == pytest.approx(0.005)


def test_participation_above_the_limit_raises() -> None:
    """Raises rather than warns. A fill the market could not absorb is fabricated, not expensive."""
    with pytest.raises(TradabilityError, match="could not have absorbed"):
        check_participation(5_000.0, 100_000.0, limit=0.01)


def test_participation_against_a_dead_session_raises() -> None:
    with pytest.raises(TradabilityError, match="no traded value"):
        check_participation(1_000.0, 0.0, limit=0.01)


def test_average_daily_value_uses_only_prior_sessions(panel: pl.DataFrame) -> None:
    """Strictly before the as-of date, so the figure is computable at decision time."""
    adv = average_daily_value(panel, "LIQUID", date(2024, 1, 15), window=21)
    assert adv == pytest.approx(100.0 * 1_000_000.0)
    # Nothing precedes the first session, so there is no history to average.
    assert average_daily_value(panel, "LIQUID", date(2024, 1, 1), window=21) == 0.0


def test_min_adv_filter_separates_liquid_from_thin(
    panel: pl.DataFrame, limits: ConstraintsConfig
) -> None:
    checker = ConstraintChecker(limits)
    assert checker.eligible(panel, "LIQUID", date(2024, 1, 20)) is True
    assert checker.eligible(panel, "THIN", date(2024, 1, 20)) is False


def test_audit_order_reports_a_clean_order_as_clean(limits: ConstraintsConfig) -> None:
    checker = ConstraintChecker(limits)
    violations = checker.audit_order(
        date(2024, 1, 10), "LIQUID", order_value=100_000.0,
        session_traded_value=100_000_000.0, fill_price=101.0, previous_close=100.0,
    )
    assert violations == []


def test_audit_order_catches_participation_and_circuit_together(
    limits: ConstraintsConfig,
) -> None:
    """Both faults are reported, not just the first. A partial diagnosis invites a partial fix."""
    checker = ConstraintChecker(limits)
    violations = checker.audit_order(
        date(2024, 1, 10), "LIQUID", order_value=50_000_000.0,
        session_traded_value=100_000_000.0, fill_price=150.0, previous_close=100.0,
    )
    kinds = {v.kind for v in violations}
    assert kinds == {"participation", "circuit"}


def test_audit_order_flags_a_session_with_no_volume(limits: ConstraintsConfig) -> None:
    checker = ConstraintChecker(limits)
    violations = checker.audit_order(
        date(2024, 1, 10), "LIQUID", order_value=1_000.0, session_traded_value=0.0,
    )
    assert [v.kind for v in violations] == ["participation"]
