"""The original, pre-optimisation ``MarketView``, frozen verbatim as an equivalence oracle.

This file exists for one reason: the PI's ruling of 2026-08-01 on engine optimisation.

    "If the vectorized engine disagrees with the approved P1 engine beyond insignificant
    floating-point differences, halt immediately. Do not choose one implementation over the other.
    Treat the disagreement as a correctness bug, identify its cause, and resolve it before any P2
    experiments are run. No P2 result may be generated from an engine whose equivalence to the
    approved P1 implementation has not been demonstrated."

Demonstrating equivalence requires something to be equivalent *to*. Every published P1 number was
produced by the implementation below, so it is kept here unmodified and
``tests/backtest/test_optimisation_equivalence.py`` runs both side by side over a coverage-chosen
set of strategies, asserting identical output.

**Nothing in production imports this.** It is an oracle, not a fallback. If it is ever edited, the
equivalence claim becomes circular and the P1 numbers lose their check — so it is not to be
"kept in sync" with the optimised version. It is a fossil, deliberately.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.common.exceptions import PointInTimeError


class ReferenceMarketView:
    """Verbatim copy of ``MarketView`` as it stood at git 4354187, before optimisation.

    The only changes are the class name and this docstring. Every method body is byte-identical to
    the implementation that produced the AlphaAudit paper's numbers.
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
