"""The interface every strategy implements, and the point-in-time view it is given.

The central design decision: a strategy is never handed the price panel. It is handed a
:class:`MarketView` that has already been truncated to the information available at its decision
moment, and which raises if asked for anything later. Lookahead therefore fails loudly at the
point of use, instead of silently improving the backtest — the strategy author cannot opt out,
because there is no object in scope that holds the future.

Signals are emitted with an explicit ``information_available_at`` timestamp. The engine checks
that stamp against the fill time and refuses any order whose information was not yet public.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from src.common.exceptions import PointInTimeError


class PanelIndex:
    """Row offsets of each trading session within a date-sorted panel.

    Built once per panel and shared by every :class:`MarketView` over it. Its whole purpose is to
    turn "the rows before ``as_of``" from a scan of the entire panel into a contiguous slice.

    Requires the panel to be sorted by ``session_date`` and **verifies it**, because a silently
    unsorted panel would make every slice below return the wrong rows — the kind of failure that
    produces plausible numbers rather than an error.
    """

    def __init__(self, panel: pl.DataFrame, date_column: str = "session_date") -> None:
        dates = panel[date_column].to_list()
        if any(dates[i] > dates[i + 1] for i in range(len(dates) - 1)):
            raise ValueError(
                "panel must be sorted by session_date to be indexed; "
                "sort it before constructing the engine"
            )
        self.panel = panel
        # Start offset of each distinct session, plus a terminal offset so every session's rows are
        # `starts[i] : starts[i+1]`.
        self.sessions: list[date] = []
        self.starts: list[int] = []
        previous: date | None = None
        for position, day in enumerate(dates):
            if day != previous:
                self.sessions.append(day)
                self.starts.append(position)
                previous = day
        self.starts.append(len(dates))

    def rows_before(self, as_of: date) -> int:
        """Number of leading rows stamped strictly before ``as_of``."""
        return self.starts[bisect_left(self.sessions, as_of)]

    def session_position(self, as_of: date) -> int:
        """Index into :attr:`sessions` of the first session at or after ``as_of``."""
        return bisect_left(self.sessions, as_of)


class MarketView:
    """Read-only window onto the panel, hard-limited to one decision date.

    Every accessor refuses data stamped on or after ``as_of``. A strategy forms its decision at
    the close of ``as_of``'s previous session and may not see ``as_of`` itself, because the engine
    fills at that session's open — reading its close would be trading on tomorrow's news.

    **On the optimisation.** The original implementation materialised the visible window in
    ``__init__`` by filtering the whole panel. Measured on the development window that cost 6.34 s
    of every 6.53 s run — the engine's own loop is 0.20 s — because a 2.29-million-row panel was
    rescanned once per session to serve a strategy that usually wanted the last twenty days.

    This version keeps every returned frame **byte-identical** and changes only how it is reached:
    the visible window is computed lazily, and the lookback accessors slice a contiguous range of
    sessions instead of scanning. ``src/backtest/_reference.py`` holds the original verbatim and
    ``tests/backtest/test_optimisation_equivalence.py`` asserts the two agree exactly, per the
    PI's ruling that no P2 result may come from an engine whose equivalence is undemonstrated.
    """

    def __init__(
        self,
        panel: pl.DataFrame,
        as_of: date,
        symbols: tuple[str, ...],
        index: PanelIndex | None = None,
    ) -> None:
        self._as_of = as_of
        self._symbols = symbols
        self._panel = panel
        # Building an index costs a pass over the panel, so the engine builds one and passes it in.
        # Constructing a view standalone (tests, notebooks) still works, just without the sharing.
        self._index = index if index is not None else PanelIndex(panel)
        self._symbol_list = list(symbols)
        self._visible_cache: pl.DataFrame | None = None

    @property
    def as_of(self) -> date:
        """The decision date. No data from this date or later is reachable."""
        return self._as_of

    @property
    def symbols(self) -> tuple[str, ...]:
        """The investable universe on this date, point-in-time."""
        return self._symbols

    @property
    def _visible(self) -> pl.DataFrame:
        """The full visible window, materialised on first use and then cached.

        Identical to the original eager computation; only the timing differs. Kept as a property
        under the original name so any code that reached for ``view._visible`` still works.
        """
        if self._visible_cache is None:
            prefix = self._panel.head(self._index.rows_before(self._as_of))
            self._visible_cache = prefix.filter(pl.col("symbol").is_in(self._symbol_list))
        return self._visible_cache

    def history(self, lookback: int | None = None) -> pl.DataFrame:
        """Visible history, optionally limited to the most recent ``lookback`` sessions."""
        if lookback is None:
            return self._visible
        if lookback <= 0:
            # `sorted(...)[-0:]` is the whole list, not an empty one. Preserving that quirk
            # deliberately: it is what the reference implementation does.
            return self._visible

        # The sessions wanted are the last `lookback` distinct dates present *after* symbol
        # filtering — not the last `lookback` calendar sessions. Those differ whenever none of the
        # held names traded on a day. Widen the slice until enough qualifying sessions are found,
        # which in practice succeeds on the first attempt.
        available = self._index.session_position(self._as_of)
        window = lookback
        while True:
            first = max(0, available - window)
            candidate = self._panel.slice(
                self._index.starts[first], self._index.starts[available] - self._index.starts[first]
            ).filter(pl.col("symbol").is_in(self._symbol_list))
            distinct = candidate["session_date"].n_unique()
            if distinct >= lookback or first == 0:
                break
            window *= 2

        sessions = sorted(candidate["session_date"].unique().to_list())[-lookback:]
        return candidate.filter(pl.col("session_date").is_in(sessions))

    def closes(self, lookback: int | None = None) -> pl.DataFrame:
        """Adjusted closes as a symbol-by-date frame, ready for cross-sectional work."""
        frame = self.history(lookback)
        return frame.pivot(on="symbol", index="session_date", values="adj_close").sort(
            "session_date"
        )

    def latest_close(self) -> dict[str, float]:
        """Most recent visible close per symbol."""
        if self._visible.is_empty():
            return {}
        newest = self._visible.sort("session_date").group_by("symbol").last()
        return dict(zip(newest["symbol"].to_list(), newest["adj_close"].to_list(), strict=True))

    def require_before(self, when: date) -> None:
        """Assert a caller-supplied timestamp really is in the past. Raises otherwise."""
        if when >= self._as_of:
            raise PointInTimeError(
                f"strategy requested data stamped {when} while deciding for {self._as_of}; "
                "only strictly earlier sessions are available"
            )


@dataclass(frozen=True)
class Signal:
    """A target portfolio, stamped with the moment its information became available.

    ``weights`` are fractions of portfolio value per symbol. They are not required to sum to one —
    the remainder is held as cash — but the engine rejects leverage above the configured cap.
    """

    information_available_at: date
    weights: dict[str, float] = field(default_factory=dict)

    def gross_exposure(self) -> float:
        return sum(abs(w) for w in self.weights.values())


class Strategy(ABC):
    """Base class for every strategy the engine can run.

    Subclasses implement :meth:`generate` and receive only a :class:`MarketView`. There is
    deliberately no hook that hands over the full panel, the future, or the test-fold boundary.
    """

    #: Human-readable economic reasoning. The semantic auditor reads this against the code, so a
    #: rationale that does not match what ``generate`` actually does is itself a finding.
    rationale: str = ""

    @abstractmethod
    def generate(self, view: MarketView) -> Signal:
        """Return the target portfolio for the session following ``view.as_of``."""

    @property
    def name(self) -> str:
        return type(self).__name__
