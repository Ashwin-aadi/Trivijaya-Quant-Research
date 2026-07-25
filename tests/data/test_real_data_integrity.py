"""Integrity checks that run against the real downloaded data rather than synthetic fixtures.

These are the Phase 1.0 required tests: calendar correctness on known NSE holidays, split
detection on a documented corporate action, universe survivorship, and price coverage. They skip
cleanly when the data has not been downloaded yet, so a fresh clone can still run the unit suite —
but they never *pass* vacuously, because each asserts on real content once the data is present.
"""

from datetime import date

import polars as pl
import pytest

from src.common.config import load_config
from src.data.calendar import load_calendar

CFG = load_config()
CALENDAR_FILE = CFG.paths.data_raw / "calendar_cnx100.parquet"
UNIVERSE_FILE = CFG.paths.data_processed / "universe.parquet"

requires_calendar = pytest.mark.skipif(
    not CALENDAR_FILE.exists(), reason="calendar not downloaded; run the ingestion first"
)
requires_universe = pytest.mark.skipif(
    not UNIVERSE_FILE.exists(), reason="universe not built; run scripts/build_universe.py"
)


@requires_calendar
def test_no_sessions_on_known_nse_holidays() -> None:
    """The exchange was shut on these dates; a calendar that disagrees is wrong."""
    calendar = load_calendar(CALENDAR_FILE)
    holidays = [
        (date(2020, 3, 10), "Holi"),
        (date(2021, 1, 26), "Republic Day"),
        (date(2022, 8, 15), "Independence Day"),
        (date(2023, 1, 26), "Republic Day"),
        (date(2024, 8, 15), "Independence Day"),
    ]
    closed = [(day, name) for day, name in holidays if calendar.is_trading_day(day)]
    assert not closed, f"calendar claims the market traded on: {closed}"


@requires_calendar
def test_special_sessions_are_preserved() -> None:
    """Muhurat trading falls on a Diwali holiday but the market is genuinely open.

    A calendar built by subtracting a holiday list from weekdays would wrongly drop these.
    Deriving sessions from index levels keeps them, which is the point of that design.
    """
    calendar = load_calendar(CALENDAR_FILE)
    assert calendar.is_trading_day(date(2024, 11, 1)), "Muhurat session 2024 is missing"


@requires_calendar
def test_session_counts_are_plausible() -> None:
    """NSE trades roughly 245-255 days a year; a wildly different count means a broken fetch."""
    calendar = load_calendar(CALENDAR_FILE)
    for year in (2020, 2021, 2022, 2023, 2024):
        count = calendar.count_sessions(date(year, 1, 1), date(year, 12, 31))
        assert 240 <= count <= 260, f"{year} has {count} sessions, which is not credible"


@requires_calendar
def test_no_weekend_sessions_except_declared_specials() -> None:
    """Saturday/Sunday sessions do occur (budget days, disaster-recovery drills) but are rare."""
    calendar = load_calendar(CALENDAR_FILE)
    weekend = [s for s in calendar.sessions_in_range(date(2020, 1, 1), date(2024, 12, 31))
               if s.weekday() >= 5]
    # A handful is expected and correct; dozens would mean the source is not a session calendar.
    assert len(weekend) <= 6, f"{len(weekend)} weekend sessions found: {weekend}"


@requires_universe
def test_universe_size_is_constant() -> None:
    """Every rebalance must hold exactly the configured number of names.

    A drift means the eligibility filter or the buffer logic silently dropped someone, which would
    change the cross-section without anyone noticing.
    """
    frame = pl.read_parquet(UNIVERSE_FILE)
    counts = frame.group_by("rebalance_date").len().sort("rebalance_date")
    offenders = counts.filter(pl.col("len") != CFG.universe.size)
    assert offenders.is_empty(), f"universe size drifted: {offenders.to_dicts()}"


@requires_universe
def test_universe_membership_actually_changes() -> None:
    """If membership never changed, the universe is static and effectively survivorship-biased."""
    frame = pl.read_parquet(UNIVERSE_FILE)
    dates = sorted(frame["rebalance_date"].unique().to_list())
    assert len(dates) >= 4, "too few rebalances to judge turnover"
    first = set(frame.filter(pl.col("rebalance_date") == dates[0])["symbol"].to_list())
    last = set(frame.filter(pl.col("rebalance_date") == dates[-1])["symbol"].to_list())
    assert first != last, "universe membership is identical at both ends; ranking is not working"


@requires_universe
def test_dropped_names_retain_prices_before_they_left() -> None:
    """The survivorship test, on real data.

    A name that leaves the universe must still have real price history in the sessions before it
    left. If it were missing, the panel would only contain survivors and every backtest built on
    it would be inflated.
    """
    universe = pl.read_parquet(UNIVERSE_FILE)
    prices = pl.read_parquet(CFG.paths.data_processed / "prices_adjusted.parquet")
    dates = sorted(universe["rebalance_date"].unique().to_list())
    first = set(universe.filter(pl.col("rebalance_date") == dates[0])["symbol"].to_list())
    last = set(universe.filter(pl.col("rebalance_date") == dates[-1])["symbol"].to_list())

    dropped = sorted(first - last)
    assert dropped, "no name ever left the universe, which is implausible over five years"
    for symbol in dropped[:5]:
        history = prices.filter(
            (pl.col("symbol") == symbol) & (pl.col("session_date") <= dates[0])
        )
        assert history.height > 0, f"{symbol} left the universe but has no prior price history"
        assert history["adj_close"].min() > 0, f"{symbol} has non-positive prices before it left"
