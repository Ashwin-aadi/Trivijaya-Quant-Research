"""Tests for the rules-based liquidity universe.

Built on a small synthetic panel so every expected outcome is checkable by hand. The point of
these tests is not that the code runs — it is that the three properties the universe claims are
actually enforced: no lookahead, buffer hysteresis, and survivorship-freedom.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.common.config import UniverseConfig
from src.common.exceptions import PointInTimeError
from src.data.calendar import TradingCalendar
from src.data.universe import (
    build_universe_history,
    rebalance_dates,
    select_universe,
)

# Small parameters so fixtures stay hand-checkable; the shape mirrors the real config.
CFG = UniverseConfig(
    method="liquidity_rank",
    size=3,
    trailing_sessions=5,
    rebalance="quarterly",
    entry_rank=3,
    exit_rank=4,
    min_listed_sessions=5,
    min_traded_fraction=0.8,
)


def weekday_sessions(start: date, n: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < n:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def panel_from(turnovers: dict[str, float], sessions: list[date]) -> pl.DataFrame:
    """Constant turnover per symbol across all sessions — ranking is then obvious by eye."""
    return pl.DataFrame(
        {
            "session_date": [s for s in sessions for _ in turnovers],
            "symbol": [sym for _ in sessions for sym in turnovers],
            "turnover_inr": [turnovers[sym] for _ in sessions for sym in turnovers],
        }
    )


def test_ranks_by_median_traded_value() -> None:
    sessions = weekday_sessions(date(2020, 1, 1), 10)
    cal = TradingCalendar(sessions)
    panel = panel_from({"BIG": 900.0, "MID": 500.0, "SMALL": 100.0, "TINY": 10.0}, sessions[:5])
    snap = select_universe(panel, cal, sessions[5], CFG)
    assert snap.symbols == ("BIG", "MID", "SMALL")   # TINY misses the size-3 cut
    assert snap.ranks["BIG"] == 1


def test_rejects_data_at_or_after_rebalance_date() -> None:
    # The core point-in-time guarantee: a panel containing the rebalance date must raise, not warn.
    sessions = weekday_sessions(date(2020, 1, 1), 10)
    cal = TradingCalendar(sessions)
    panel = panel_from({"A": 5.0, "B": 4.0, "C": 3.0}, sessions[:7])
    with pytest.raises(PointInTimeError):
        select_universe(panel, cal, sessions[6], CFG)


def test_buffer_protects_incumbent_but_not_newcomer() -> None:
    sessions = weekday_sessions(date(2020, 1, 1), 10)
    cal = TradingCalendar(sessions)
    # DRIFTER sits at rank 4: past entry_rank (3) but inside exit_rank (4).
    panel = panel_from(
        {"A": 900.0, "B": 800.0, "C": 700.0, "DRIFTER": 600.0, "E": 500.0}, sessions[:5]
    )
    as_of = sessions[5]

    fresh = select_universe(panel, cal, as_of, CFG, incumbents=frozenset())
    assert "DRIFTER" not in fresh.symbols        # cannot enter at rank 4

    held = select_universe(panel, cal, as_of, CFG, incumbents=frozenset({"DRIFTER"}))
    assert "DRIFTER" in held.symbols             # but an incumbent is not churned out


def test_delisted_stock_stays_in_earlier_universes() -> None:
    """The survivorship property, stated as a test.

    A stock that trades heavily and then disappears must remain in the universes covering the
    period when it was trading. If it vanished from history, the backtest would only ever see
    survivors and every downstream result would be inflated.
    """
    # Long enough to span several quarterly rebalances, so there is a "before" and an "after".
    sessions = weekday_sessions(date(2020, 1, 1), 160)
    cal = TradingCalendar(sessions)
    survivors = {"A": 900.0, "B": 800.0, "C": 300.0}

    cutoff = 40
    early = panel_from({**survivors, "DOOMED": 850.0}, sessions[:cutoff])  # trades, ranks high
    late = panel_from(survivors, sessions[cutoff:])                        # then stops existing
    panel = pl.concat([early, late])

    snaps = build_universe_history(panel, cal, sessions[10], sessions[-1], CFG)
    assert len(snaps) >= 2, "test needs at least one rebalance before and one after the delisting"

    assert "DOOMED" in set(snaps[0].symbols), \
        "a stock that later delisted was erased from its own past"
    # And it correctly drops out once it stops trading, without being retroactively removed.
    assert "DOOMED" not in set(snaps[-1].symbols)


def test_rebalance_dates_quarterly_and_monthly() -> None:
    sessions = weekday_sessions(date(2020, 1, 1), 260)
    cal = TradingCalendar(sessions)
    quarterly = rebalance_dates(cal, sessions[0], sessions[-1], "quarterly")
    monthly = rebalance_dates(cal, sessions[0], sessions[-1], "monthly")
    assert len(quarterly) == 4                      # one year of weekdays -> 4 quarters
    assert len(monthly) == 12
    assert quarterly[0] == sessions[0]
    assert all(cal.is_trading_day(d) for d in quarterly)


def test_unknown_frequency_rejected() -> None:
    sessions = weekday_sessions(date(2020, 1, 1), 30)
    cal = TradingCalendar(sessions)
    with pytest.raises(ValueError, match="unsupported rebalance frequency"):
        rebalance_dates(cal, sessions[0], sessions[-1], "fortnightly")
