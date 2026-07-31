"""The backtest engine: turns signals into fills, positions, and a PnL record.

Timing rule, enforced rather than trusted: a signal formed from session *t*'s close is filled at
session *t+1*'s **open**. The engine constructs each :class:`MarketView` already truncated to
before the fill session, and then re-checks every order's ``information_available_at`` against its
fill date. A strategy that manufactures a stale-but-forward timestamp still raises
:class:`PointInTimeError` — the check is not a formality, it is the last line of defence.

Correctness is preferred to speed throughout. The loop is per-session and explicit, because a
vectorised formulation makes it far easier to shift an array by the wrong sign and never notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy
from src.common.exceptions import PointInTimeError
from src.common.log import get_logger
from src.costs.india import CostModel
from src.data.calendar import TradingCalendar

_log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Per-session record of everything the engine did."""

    dates: list[date] = field(default_factory=list)
    equity: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    gross_exposure: list[float] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    # Gross return before transaction costs, kept alongside the net figure so the cost drag is a
    # measured quantity rather than a difference between two separate runs.
    gross_returns: list[float] = field(default_factory=list)
    # Orders that would have consumed more than the configured share of a session's traded value.
    # Recorded rather than raised so a whole corpus completes and the pattern can be reported; an
    # engine that aborts on the first breach tells you nothing about how common breaches are.
    participation_breaches: list[tuple[date, str, float]] = field(default_factory=list)
    # The session on which equity reached zero, if it did. A strategy that loses more than the
    # account has is bankrupt, and the sessions after that point are not a worse result — they are
    # not a result at all, because there is nothing left to trade.
    ruined_on: date | None = None
    # Sessions where a held name sat inside a known-artifacts window. Not an error: a marker so an
    # apparent edge that depends on flagged data can be recognised instead of trusted.
    flagged_sessions: list[date] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "session_date": self.dates,
                "equity": self.equity,
                "return": self.returns,
                "gross_exposure": self.gross_exposure,
                "turnover": self.turnover,
                "cost": self.costs,
                "gross_return": self.gross_returns,
            }
        )


class BacktestEngine:
    """Runs one strategy over a date range with point-in-time discipline."""

    def __init__(
        self,
        panel: pl.DataFrame,
        calendar: TradingCalendar,
        universe: pl.DataFrame,
        *,
        max_gross_exposure: float = 1.0,
        artifact_register: pl.DataFrame | None = None,
        cost_model: CostModel | None = None,
    ) -> None:
        self._panel = panel
        self._calendar = calendar
        self._universe = universe
        self._max_gross = max_gross_exposure
        self._register = artifact_register
        self._costs = cost_model

        # Opens, closes and traded values are pulled into dicts once. Per-session dataframe
        # filtering inside the loop dominated runtime and bought nothing in clarity.
        self._open: dict[tuple[str, date], float] = {}
        self._close: dict[tuple[str, date], float] = {}
        self._traded_value: dict[tuple[str, date], float] = {}
        has_turnover = "turnover_inr" in panel.columns
        columns = ["symbol", "session_date", "adj_open", "adj_close"]
        if has_turnover:
            columns.append("turnover_inr")
        for row in panel.select(columns).iter_rows(named=True):
            key = (row["symbol"], row["session_date"])
            self._open[key] = row["adj_open"]
            self._close[key] = row["adj_close"]
            if has_turnover:
                self._traded_value[key] = row["turnover_inr"] or 0.0

    def _universe_on(self, day: date) -> tuple[str, ...]:
        """Constituents effective on ``day``: the most recent rebalance at or before it."""
        eligible = self._universe.filter(pl.col("rebalance_date") <= day)
        if eligible.is_empty():
            return ()
        latest = eligible["rebalance_date"].max()
        return tuple(eligible.filter(pl.col("rebalance_date") == latest)["symbol"].to_list())

    def _flagged(self, symbol: str, day: date) -> bool:
        if self._register is None:
            return False
        hit = self._register.filter(
            (pl.col("symbol") == symbol)
            & (pl.col("start_date") <= day)
            & (pl.col("end_date") >= day)
        )
        return hit.height > 0

    def run(
        self,
        strategy: Strategy,
        start: date,
        end: date,
        initial_equity: float = 1_000_000.0,
        max_participation_rate: float = 0.01,
    ) -> BacktestResult:
        """Execute ``strategy`` across every session in [start, end].

        Transaction costs come from the :class:`CostModel` handed to the constructor, charged leg
        by leg at the rates in force on the fill session. With no cost model the run is **gross**,
        and the cost column is zero — that is a deliberate configuration, not a silent default:
        the gross and net figures are both wanted, and reporting one as the other is the failure
        mode this phase exists to fix.
        """
        sessions = self._calendar.sessions_in_range(start, end)
        if len(sessions) < 2:
            raise ValueError(f"need at least two sessions between {start} and {end}")

        result = BacktestResult()
        equity = initial_equity
        holdings: dict[str, float] = {}          # symbol -> fraction of equity

        # Decide at session i-1's close, fill at session i's open. The loop starts at 1 so the
        # first fill always has a prior session to have been decided on.
        for index in range(1, len(sessions)):
            fill_session = sessions[index]
            decision_session = sessions[index - 1]

            symbols = self._universe_on(decision_session)
            if not symbols:
                continue

            view = MarketView(self._panel, as_of=fill_session, symbols=symbols)
            signal = strategy.generate(view)
            self._validate(signal, decision_session, fill_session, strategy)

            target = self._clamp(signal.weights, symbols)
            turnover = sum(abs(target.get(s, 0.0) - holdings.get(s, 0.0))
                           for s in set(target) | set(holdings))
            cost = self._charge(
                target, holdings, equity, fill_session, result, max_participation_rate
            )

            # Return accrual, split by how each rupee of the position came to be held.
            #
            # A position carried over from the previous session was already owned at last night's
            # close, so it earns the full close-to-close move including the overnight gap. Only the
            # portion bought *this morning* earns from the open, because that is the price it was
            # actually acquired at.
            #
            # Collapsing both cases into open-to-close would silently liquidate the book every
            # evening and re-buy it every morning, discarding every overnight return in the sample.
            # In this market that is not a small approximation — it is most of the equity premium,
            # and it makes any long-horizon strategy look structurally unprofitable.
            period_return = 0.0
            for symbol, weight in target.items():
                open_px = self._open.get((symbol, fill_session))
                close_px = self._close.get((symbol, fill_session))
                prior_close = self._close.get((symbol, decision_session))
                if not open_px or not close_px or open_px <= 0:
                    continue

                previous_weight = holdings.get(symbol, 0.0)
                # Weight retained from yesterday, only where the sign is unchanged.
                carried = (min(previous_weight, weight) if weight >= 0
                           else max(previous_weight, weight)) if previous_weight * weight > 0 \
                    else 0.0
                newly_bought = weight - carried

                if carried and prior_close and prior_close > 0:
                    period_return += carried * (close_px / prior_close - 1.0)
                else:
                    newly_bought = weight        # nothing carried; the whole position is new today
                period_return += newly_bought * (close_px / open_px - 1.0)

                if self._flagged(symbol, fill_session):
                    result.flagged_sessions.append(fill_session)

            net_return = period_return - cost
            equity *= 1.0 + net_return
            holdings = target

            result.dates.append(fill_session)
            result.equity.append(equity)
            result.returns.append(net_return)
            result.gross_returns.append(period_return)
            result.gross_exposure.append(sum(abs(w) for w in target.values()))
            result.turnover.append(turnover)
            result.costs.append(cost)

            # Bankruptcy is terminal, and it has to be represented rather than run through. Once
            # equity is non-positive there is no capital to trade and no meaningful return to
            # compute: the next session's cost would be charged against a negative notional, which
            # is arithmetic, not economics. Surfaced by the net positive control, where a strategy
            # rotating a hundred-name book every session was consumed by depository charges.
            if equity <= 0.0:
                result.ruined_on = fill_session
                _log.warning("%s: ruined on %s after %d sessions",
                             strategy.name, fill_session, len(result.dates))
                break

        _log.info("%s: %d sessions, final equity %.2f (%d flagged sessions)",
                  strategy.name, len(result.dates), equity, len(set(result.flagged_sessions)))
        return result

    def _charge(
        self,
        target: dict[str, float],
        holdings: dict[str, float],
        equity: float,
        fill_session: date,
        result: BacktestResult,
        max_participation_rate: float,
    ) -> float:
        """Transaction cost of moving from ``holdings`` to ``target``, as a fraction of equity.

        Charged **per name and per side**, never as a single blended round-trip rate. The two legs
        are not symmetric — stamp duty falls on the buy alone, the depository charge on the sell
        alone — so a strategy that rotates into new names pays a different amount from one that
        rotates out, and a blended figure gets both wrong.

        Rates are those in force on ``fill_session``, not today's.
        """
        if self._costs is None:
            return 0.0

        total = 0.0
        for symbol in set(target) | set(holdings):
            delta = target.get(symbol, 0.0) - holdings.get(symbol, 0.0)
            if delta == 0.0:
                continue
            rupees = abs(delta) * equity
            traded_value = self._traded_value.get((symbol, fill_session), 0.0)

            if traded_value > 0:
                participation = rupees / traded_value
                if participation > max_participation_rate:
                    result.participation_breaches.append((fill_session, symbol, participation))

            # One scrip per name sold: the depository charge is levied per symbol, not per rupee,
            # which is what makes it bite hardest on the small positions a diversified book holds.
            total += self._costs.execute(
                rupees,
                "buy" if delta > 0 else "sell",
                day_traded_value=traded_value,
                n_scrips=0 if delta > 0 else 1,
                on=fill_session,
            ).total

        return total / equity if equity else 0.0

    def _validate(
        self,
        signal: Signal,
        decision_session: date,
        fill_session: date,
        strategy: Strategy,
    ) -> None:
        """Reject any order whose information was not available before the fill."""
        if signal.information_available_at >= fill_session:
            raise PointInTimeError(
                f"{strategy.name} produced a signal stamped "
                f"{signal.information_available_at} to be filled at {fill_session}; the stamp must "
                "be strictly earlier than the fill session"
            )
        if signal.information_available_at > decision_session:
            raise PointInTimeError(
                f"{strategy.name} claims information from "
                f"{signal.information_available_at}, later than the decision session "
                f"{decision_session}"
            )

    def _clamp(self, weights: dict[str, float], symbols: tuple[str, ...]) -> dict[str, float]:
        """Drop non-universe names and scale back any position breaching the exposure cap."""
        allowed = {s: w for s, w in weights.items() if s in symbols and w == w}  # w==w drops NaN
        gross = sum(abs(w) for w in allowed.values())
        if gross > self._max_gross and gross > 0:
            scale = self._max_gross / gross
            allowed = {s: w * scale for s, w in allowed.items()}
        return allowed
