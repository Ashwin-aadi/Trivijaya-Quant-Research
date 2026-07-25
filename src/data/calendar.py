"""NSE trading calendar: which dates the exchange was open, and arithmetic over them.

Split into two concerns on purpose:

* :class:`TradingCalendar` is pure logic over a set of trading days — membership, next/previous
  session, counting, and shifting by N sessions. It knows nothing about where the days came from,
  so it is fully testable without a network call and reusable for any exchange.
* :func:`sessions_from_weekdays` builds a day set from a weekend rule plus an *injected* holiday
  list. The holiday list is deliberately not hard-coded here: an accurate 2015-present NSE holiday
  history has to come from a citable source (chosen at Checkpoint 1.0), not from memory.

The backtester leans on ``shift`` for its core timing rule — a signal formed at a session's close
may fill no earlier than the next session's open — and on ``require_trading_day`` to make trading
on a closed day impossible rather than merely discouraged.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.common.exceptions import DataIntegrityError, LabError
from src.common.io import read_parquet, write_raw_parquet
from src.common.log import get_logger

_log = get_logger(__name__)

# Monday=0 ... Sunday=6. NSE is closed on the weekend; intraweek closures are holidays.
_WEEKEND = {5, 6}


class CalendarError(LabError):
    """A calendar query was made outside the calendar's known range, or the calendar is empty."""


class TradingCalendar:
    """An ordered set of trading sessions with date arithmetic defined over it."""

    def __init__(self, sessions: Iterable[date], name: str = "NSE") -> None:
        # De-dupe and sort once; downstream queries assume a strictly increasing sequence.
        unique = sorted(set(sessions))
        if not unique:
            raise CalendarError("cannot build a TradingCalendar with zero sessions")
        self._sessions: tuple[date, ...] = tuple(unique)
        self._session_set: frozenset[date] = frozenset(unique)
        self.name = name

    # --- basic properties --------------------------------------------------
    @property
    def first_session(self) -> date:
        return self._sessions[0]

    @property
    def last_session(self) -> date:
        return self._sessions[-1]

    @property
    def n_sessions(self) -> int:
        return len(self._sessions)

    # --- membership --------------------------------------------------------
    def is_trading_day(self, day: date) -> bool:
        """True iff the exchange was open on ``day``."""
        return day in self._session_set

    def require_trading_day(self, day: date) -> None:
        """Raise if ``day`` is not a trading day. Used to make closed-day trades impossible."""
        if day not in self._session_set:
            raise CalendarError(f"{day} is not an {self.name} trading day")

    def _check_in_range(self, day: date) -> None:
        if day < self.first_session or day > self.last_session:
            raise CalendarError(
                f"{day} is outside calendar range "
                f"[{self.first_session}, {self.last_session}]"
            )

    # --- navigation --------------------------------------------------------
    def next_session(self, day: date, *, strictly_after: bool = True) -> date:
        """The first trading day after ``day`` (or on/after it if ``strictly_after`` is False)."""
        self._check_in_range(day)
        idx = bisect.bisect_right(self._sessions, day) if strictly_after \
            else bisect.bisect_left(self._sessions, day)
        if idx >= len(self._sessions):
            raise CalendarError(f"no trading day after {day} within calendar range")
        return self._sessions[idx]

    def previous_session(self, day: date, *, strictly_before: bool = True) -> date:
        """The last trading day before ``day`` (or on/before it if ``strictly_before`` is False)."""
        self._check_in_range(day)
        idx = bisect.bisect_left(self._sessions, day) if strictly_before \
            else bisect.bisect_right(self._sessions, day)
        if idx == 0:
            raise CalendarError(f"no trading day before {day} within calendar range")
        return self._sessions[idx - 1]

    def shift(self, day: date, n: int) -> date:
        """Return the session ``n`` trading days from ``day`` (n<0 goes back). ``day`` must trade.

        This is the primitive the backtester uses for next-open execution (``shift(day, 1)``) and
        for embargo windows around a test fold, so it insists ``day`` itself is a trading day —
        shifting from an ambiguous non-session start is a bug waiting to happen.
        """
        self.require_trading_day(day)
        idx = bisect.bisect_left(self._sessions, day) + n
        if idx < 0 or idx >= len(self._sessions):
            raise CalendarError(f"shifting {day} by {n} sessions falls outside calendar range")
        return self._sessions[idx]

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        """All trading days in the inclusive range [start, end]."""
        if start > end:
            raise CalendarError(f"start {start} is after end {end}")
        lo = bisect.bisect_left(self._sessions, start)
        hi = bisect.bisect_right(self._sessions, end)
        return list(self._sessions[lo:hi])

    def count_sessions(self, start: date, end: date) -> int:
        """Number of trading days in [start, end] inclusive."""
        return len(self.sessions_in_range(start, end))


def fetch_index_sessions(
    symbol: str,
    start: date,
    end: date,
    raw_path: Path,
) -> pl.DataFrame:
    """Download index level history and cache it immutably as the calendar's source of truth.

    The exchange published a level for this index on exactly the days it was open, so the dates
    present in this series *are* the trading sessions — no hand-maintained holiday list, and no
    dependence on the stock price source, which keeps "no price rows on a holiday" an independent
    check rather than a tautology.

    Special sessions (Muhurat trading, budget-day Saturdays, disaster-recovery sessions) carry an
    index level like any other day and are therefore preserved automatically. Nothing here filters
    on weekday, precisely so those days are not silently discarded as anomalies.
    """
    if raw_path.exists():
        _log.info("index history already cached at %s", raw_path)
        return read_parquet(raw_path)

    import yfinance as yf  # imported lazily so the pure calendar logic needs no network stack

    frame = yf.Ticker(symbol).history(
        start=start.isoformat(), end=end.isoformat(), auto_adjust=False
    )
    if frame.empty:
        raise DataIntegrityError(
            f"index history for {symbol} came back empty for {start}..{end}; "
            "refusing to build a calendar from nothing"
        )
    sessions = pl.DataFrame(
        {
            "session_date": [ts.date() for ts in frame.index],
            "close": frame["Close"].to_list(),
        }
    )
    write_raw_parquet(
        sessions,
        raw_path,
        source_url=f"yfinance:{symbol}",
        extra={
            "symbol": symbol,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "purpose": "trading session calendar",
        },
    )
    _log.info("cached %d sessions for %s to %s", sessions.height, symbol, raw_path)
    return sessions


def load_calendar(raw_path: Path, name: str = "NSE") -> TradingCalendar:
    """Build a TradingCalendar from a cached index-history file."""
    frame = read_parquet(raw_path)
    return TradingCalendar(frame["session_date"].to_list(), name=name)


def sessions_from_weekdays(
    start: date,
    end: date,
    holidays: Iterable[date],
) -> list[date]:
    """Build a trading-day list as weekdays in [start, end] minus an injected holiday set.

    The holiday set must come from a citable source; this function does not invent one. It exists
    so that, once that source is fixed, constructing the calendar is a one-liner and the weekend
    rule lives in exactly one place.
    """
    holiday_set = set(holidays)
    days: list[date] = []
    cursor = start
    step = timedelta(days=1)
    while cursor <= end:
        if cursor.weekday() not in _WEEKEND and cursor not in holiday_set:
            days.append(cursor)
        cursor += step
    return days
