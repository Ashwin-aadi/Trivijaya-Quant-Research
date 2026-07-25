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
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from src.common.exceptions import PointInTimeError


class MarketView:
    """Read-only window onto the panel, hard-limited to one decision date.

    Every accessor refuses data stamped on or after ``as_of``. A strategy forms its decision at
    the close of ``as_of``'s previous session and may not see ``as_of`` itself, because the engine
    fills at that session's open — reading its close would be trading on tomorrow's news.
    """

    def __init__(self, panel: pl.DataFrame, as_of: date, symbols: tuple[str, ...]) -> None:
        self._as_of = as_of
        self._symbols = symbols
        # Truncation happens once, here. Nothing downstream can widen it back out.
        self._visible = panel.filter(
            (pl.col("session_date") < as_of) & (pl.col("symbol").is_in(list(symbols)))
        )

    @property
    def as_of(self) -> date:
        """The decision date. No data from this date or later is reachable."""
        return self._as_of

    @property
    def symbols(self) -> tuple[str, ...]:
        """The investable universe on this date, point-in-time."""
        return self._symbols

    def history(self, lookback: int | None = None) -> pl.DataFrame:
        """Visible history, optionally limited to the most recent ``lookback`` sessions."""
        if lookback is None:
            return self._visible
        sessions = sorted(self._visible["session_date"].unique().to_list())[-lookback:]
        return self._visible.filter(pl.col("session_date").is_in(sessions))

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
