"""Tradability constraints: whether a backtested trade could actually have been executed.

A cost model answers *what would this trade have cost*. This module answers the prior question —
*could this trade have happened at all*. A backtest that fills an order the market could not have
absorbed is not expensive, it is fictional, and no cost adjustment repairs it.

Three limits, each for a different reason:

**Circuit limits.** NSE applies price bands, and no trade prints outside them. A strategy whose
signal fires on a stock locked at its upper circuit did not buy it — nobody was selling.

**Minimum average daily traded value.** A name that trades a few lakh rupees a day cannot absorb an
institutional position at any price. Filtering on ADV is what keeps a backtest's universe
investable rather than merely listed.

**Maximum participation.** The hard one, and the one most often skipped. If a strategy's order is a
large fraction of a day's volume, the price it gets is not the price in the data — it *is* the
trade. The charter requires a hard assertion, not a warning, because a silent violation produces a
result that looks excellent and could never be realised.

Participation is checked against the traded value *of the session being filled*, which is known
only after the fact. That is deliberate and is not lookahead: it is a feasibility audit run after a
backtest, answering "was this achievable", not a signal input. Nothing here may be fed back into a
trading decision, and nothing in this module returns a value a strategy could condition on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.common.config import ConstraintsConfig
from src.common.exceptions import DataIntegrityError


class TradabilityError(DataIntegrityError):
    """Raised when a backtest fills an order the market could not have absorbed."""


@dataclass(frozen=True)
class Violation:
    """One order that could not have been executed as backtested."""

    session: date
    symbol: str
    kind: str            # "participation" | "circuit" | "min_adv"
    detail: str

    def __str__(self) -> str:
        return f"{self.session} {self.symbol}: {self.kind} - {self.detail}"


def within_circuit(
    fill_price: float, previous_close: float, band: float
) -> bool:
    """Whether ``fill_price`` lies inside the price band around the previous close.

    Equality with the band edge counts as inside: a stock locked *at* its circuit can still trade
    at that price, it simply cannot trade beyond it.
    """
    if previous_close <= 0:
        return False
    return abs(fill_price / previous_close - 1.0) <= band + 1e-12


def average_daily_value(
    panel: pl.DataFrame, symbol: str, as_of: date, window: int
) -> float:
    """Mean rupee traded value for ``symbol`` over the ``window`` sessions before ``as_of``.

    Strictly before, so this is computable at decision time and carries no information from the
    session being traded.
    """
    rows = (
        panel.filter((pl.col("symbol") == symbol) & (pl.col("session_date") < as_of))
        .sort("session_date")
        .tail(window)
    )
    if rows.is_empty():
        return 0.0
    if "traded_value" in rows.columns:
        return _as_float(rows["traded_value"].mean())
    # Fall back to close * volume where the exchange's own turnover column is absent. Noted rather
    # than silent: the exchange figure is authoritative and this reconstruction is not identical.
    return _as_float((rows["close"] * rows["volume"]).mean())


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float, treating anything non-numeric as no data.

    ``Series.mean()`` is typed as a union spanning dates and strings because the same method serves
    every dtype. Narrowing here rather than casting keeps a wrongly-typed column from silently
    becoming a number.
    """
    return float(value) if isinstance(value, int | float) else 0.0


def check_participation(
    order_value: float, session_traded_value: float, limit: float
) -> float:
    """Return the participation rate, raising if it exceeds ``limit``.

    Raises rather than warns, per the charter. A strategy consuming more of a session than the
    limit allows has not been penalised by an inadequate cost model — it has been credited with a
    fill that did not exist, and that is not a costing error but a fabricated result.
    """
    if session_traded_value <= 0:
        raise TradabilityError(
            f"order of {order_value:,.0f} against a session with no traded value; "
            "the fill could not have occurred"
        )
    rate = order_value / session_traded_value
    if rate > limit:
        raise TradabilityError(
            f"order of {order_value:,.0f} is {rate:.2%} of the session's traded value of "
            f"{session_traded_value:,.0f}, above the {limit:.2%} limit. The backtest filled a "
            "trade the market could not have absorbed."
        )
    return rate


class ConstraintChecker:
    """Audits executed orders against the tradability limits in ``config.yaml``."""

    def __init__(self, config: ConstraintsConfig) -> None:
        self._cfg = config

    def eligible(self, panel: pl.DataFrame, symbol: str, as_of: date) -> bool:
        """Whether ``symbol`` clears the minimum-ADV filter as at ``as_of``."""
        adv = average_daily_value(panel, symbol, as_of, self._cfg.adv_window_sessions)
        return adv >= self._cfg.min_adv_rupees

    def audit_order(
        self,
        session: date,
        symbol: str,
        order_value: float,
        session_traded_value: float,
        fill_price: float | None = None,
        previous_close: float | None = None,
    ) -> list[Violation]:
        """Every way this order was infeasible. Empty means it could have been executed.

        Collects rather than raises so a whole backtest can be audited in one pass and the full
        pattern reported; :func:`check_participation` is the raising form for use inside an engine.
        """
        violations: list[Violation] = []

        if session_traded_value <= 0:
            violations.append(Violation(session, symbol, "participation",
                                        "session had no traded value"))
        else:
            rate = order_value / session_traded_value
            if rate > self._cfg.max_participation_rate:
                violations.append(Violation(
                    session, symbol, "participation",
                    f"{rate:.2%} of session volume, limit {self._cfg.max_participation_rate:.2%}",
                ))

        if fill_price is not None and previous_close is not None and \
                not within_circuit(fill_price, previous_close, self._cfg.circuit_band):
            move = fill_price / previous_close - 1.0 if previous_close else float("nan")
            violations.append(Violation(
                session, symbol, "circuit",
                f"fill at {fill_price:.2f} is {move:+.2%} from previous close "
                f"{previous_close:.2f}, outside the {self._cfg.circuit_band:.0%} band",
            ))

        return violations
