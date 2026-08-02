"""Tests for the backtest engine and the point-in-time market view.

Everything here runs against a synthetic panel built in this file. That is deliberate: these tests
must pass on a fresh clone with an empty ``data/`` directory, and the arithmetic has to be
checkable by hand. A price series that rises by exactly the same percentage every session is not
realistic, and is not meant to be — it makes the expected equity curve a single closed-form
expression, so a wrong answer is a wrong answer rather than a plausible one.

The engine's central guarantee is that a signal may not be filled before its information existed.
That guarantee is tested from both sides: a strategy stamping the fill session itself must raise,
and so must one stamping a date that falls in a weekend gap between the decision session and the
fill. A rule that only rejects the obvious case is not a rule.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MarketView, Signal, Strategy
from src.common.config import load_config
from src.common.exceptions import PointInTimeError
from src.costs.india import CostModel
from src.data.calendar import TradingCalendar, sessions_from_weekdays

SYMBOLS: tuple[str, ...] = ("AAA", "BBB", "CCC")
INITIAL_EQUITY = 1_000_000.0
DAILY_GAIN = 0.01


# --- synthetic fixtures ---------------------------------------------------------


def make_sessions(count: int = 30) -> list[date]:
    """``count`` consecutive NSE-shaped weekday sessions starting Monday 2024-01-01.

    Weekends are excluded, which matters: the gap between a Friday decision and a Monday fill is
    what makes the "stamp lands between decision and fill" test constructible at all.
    """
    weekdays = sessions_from_weekdays(
        start=date(2024, 1, 1),
        end=date(2024, 1, 1) + timedelta(days=90),
        holidays=[],
    )
    return weekdays[:count]


def make_calendar(sessions: list[date]) -> TradingCalendar:
    return TradingCalendar(sessions, name="TEST")


def make_panel(sessions: list[date], daily_gain: float = DAILY_GAIN) -> pl.DataFrame:
    """A panel where every symbol closes ``daily_gain`` above its own open, every session.

    The engine earns the open-to-close move of the fill session only, so this construction makes
    the per-session portfolio return exactly ``daily_gain`` for any fully invested portfolio,
    whatever the weights. Symbols start at different price levels so a bug that keys on price
    rather than on symbol has somewhere to show itself.
    """
    symbols: list[str] = []
    dates: list[date] = []
    opens: list[float] = []
    closes: list[float] = []
    for index, day in enumerate(sessions):
        for offset, symbol in enumerate(SYMBOLS):
            open_price = (100.0 + 10.0 * offset) * (1.0 + daily_gain) ** index
            symbols.append(symbol)
            dates.append(day)
            opens.append(open_price)
            closes.append(open_price * (1.0 + daily_gain))
    return pl.DataFrame(
        {
            "symbol": symbols,
            "session_date": dates,
            "adj_open": opens,
            "adj_close": closes,
            # Deep enough that participation is negligible, so the cost test measures statutory
            # charges rather than the slippage cap. A thin-tape case is tested separately.
            "turnover_inr": [1_000_000_000.0] * len(symbols),
        }
    )


def make_universe(entries: list[tuple[date, tuple[str, ...]]]) -> pl.DataFrame:
    """A point-in-time universe frame: one block of constituents per rebalance date."""
    rebalance_dates: list[date] = []
    symbols: list[str] = []
    ranks: list[int] = []
    for rebalance_date, members in entries:
        for position, symbol in enumerate(members, start=1):
            rebalance_dates.append(rebalance_date)
            symbols.append(symbol)
            ranks.append(position)
    return pl.DataFrame(
        {"rebalance_date": rebalance_dates, "symbol": symbols, "rank": ranks}
    )


def make_register(entries: list[tuple[str, date, date]]) -> pl.DataFrame:
    """A known-artifacts register in the shape ``src.data.artifacts`` writes."""
    return pl.DataFrame(
        {
            "symbol": [symbol for symbol, _, _ in entries],
            "start_date": [start for _, start, _ in entries],
            "end_date": [end for _, _, end in entries],
            "reason": ["price_jump"] * len(entries),
            "detail": ["synthetic flag for testing"] * len(entries),
        }
    )


def latest_visible_session(view: MarketView) -> date:
    """The most recent session the view will serve — the strategy's honest decision date."""
    history = view.history()
    if history.is_empty():
        raise AssertionError("view has no visible history; the fixture panel is wrong")
    latest = history["session_date"].max()
    assert isinstance(latest, date)
    return latest


# --- strategies used by the tests -----------------------------------------------


class CashOnly(Strategy):
    """Holds nothing at all. The engine's arithmetic identity."""

    rationale = "Hold cash. No position is taken, so no return and no cost can be earned."

    def generate(self, view: MarketView) -> Signal:
        return Signal(information_available_at=latest_visible_session(view), weights={})


class EqualWeightHold(Strategy):
    """Equal weight across whatever the point-in-time universe contains, rebalanced daily."""

    rationale = "Own the index. Equal weights across current constituents, held throughout."

    def generate(self, view: MarketView) -> Signal:
        weight = 1.0 / len(view.symbols)
        return Signal(
            information_available_at=latest_visible_session(view),
            weights=dict.fromkeys(view.symbols, weight),
        )


class FixedWeights(Strategy):
    """Requests a caller-supplied weight book regardless of what the view shows.

    Used to probe the engine's clamping and universe filtering, both of which must hold whatever
    the strategy asks for.
    """

    rationale = "Fixed target book, used to exercise the engine's exposure and universe controls."

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=latest_visible_session(view),
            weights=dict(self._weights),
        )


class ClaimsFillSession(Strategy):
    """Stamps its signal with the fill session itself — trading on the bar it is about to trade."""

    rationale = "Claims to know the fill session's information at the moment of deciding."

    def generate(self, view: MarketView) -> Signal:
        return Signal(information_available_at=view.as_of, weights={"AAA": 1.0})


class ClaimsFuture(Strategy):
    """Stamps its signal with a date after the fill session."""

    rationale = "Claims information from after the fill; the crudest possible lookahead."

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=view.as_of + timedelta(days=1),
            weights={"AAA": 1.0},
        )


class ClaimsCalendarDayBeforeFill(Strategy):
    """Stamps the calendar day before the fill, which on a Monday fill is a Sunday.

    Earlier than the fill, so it passes the first check, but later than Friday's decision session,
    so it must still be refused. This is the case a naive "is the stamp before the fill?" test
    would wave through.
    """

    rationale = "Claims information from the calendar day before the fill, not the prior session."

    def generate(self, view: MarketView) -> Signal:
        return Signal(
            information_available_at=view.as_of - timedelta(days=1),
            weights={"AAA": 1.0},
        )


class RecordingStrategy(Strategy):
    """Holds nothing but keeps every view it was handed, so the tests can inspect them."""

    rationale = "Instrumentation only. Records the views the engine constructs and holds cash."

    def __init__(self) -> None:
        self.views: list[MarketView] = []

    def generate(self, view: MarketView) -> Signal:
        self.views.append(view)
        return Signal(information_available_at=latest_visible_session(view), weights={})


def build_engine(
    sessions: list[date],
    *,
    universe: pl.DataFrame | None = None,
    max_gross_exposure: float = 1.0,
    artifact_register: pl.DataFrame | None = None,
    daily_gain: float = DAILY_GAIN,
    record_positions: bool = False,
) -> BacktestEngine:
    """An engine wired to the synthetic panel, with the full symbol set live from session zero."""
    panel = make_panel(sessions, daily_gain=daily_gain)
    if universe is None:
        universe = make_universe([(sessions[0], SYMBOLS)])
    return BacktestEngine(
        panel,
        make_calendar(sessions),
        universe,
        max_gross_exposure=max_gross_exposure,
        artifact_register=artifact_register,
        record_positions=record_positions,
    )


# --- the arithmetic identity ----------------------------------------------------


def test_cash_only_strategy_returns_exactly_zero() -> None:
    """Holding nothing must produce exactly 0.0 every session, and exactly 0.0 of cost.

    Asserted with ``==`` rather than ``approx`` on purpose. Any drift here is the engine inventing
    a return out of an empty portfolio, and a tolerance would hide precisely the bug worth finding.
    """
    sessions = make_sessions()
    engine = build_engine(sessions)
    result = engine.run(CashOnly(), sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY)

    # Decide at session i-1's close, fill at session i's open: one fill per session but the first.
    assert len(result.dates) == len(sessions) - 1
    assert result.dates[0] == sessions[1]
    assert result.dates[-1] == sessions[-1]

    assert all(value == 0.0 for value in result.returns)
    assert all(value == 0.0 for value in result.costs)
    assert all(value == 0.0 for value in result.turnover)
    assert all(value == 0.0 for value in result.gross_exposure)
    assert all(value == INITIAL_EQUITY for value in result.equity)


def test_buy_and_hold_compounds_the_expected_equity_curve() -> None:
    """A fully invested book on a panel that gains 1% intraday every session.

    29 fills at exactly +1% each, so final equity is 1,000,000 x 1.01^29 = 1,334,504 to the rupee.
    Turnover is 1.0 on the first fill (cash into a full book) and zero thereafter, because the
    target never changes — a buy-and-hold that keeps trading would show up here immediately.
    """
    sessions = make_sessions()
    engine = build_engine(sessions)
    result = engine.run(
        EqualWeightHold(), sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY
    )

    n_fills = len(sessions) - 1
    assert len(result.dates) == n_fills

    expected_equity = INITIAL_EQUITY * (1.0 + DAILY_GAIN) ** n_fills
    assert result.equity[-1] == pytest.approx(expected_equity, rel=1e-9)
    assert all(value == pytest.approx(DAILY_GAIN, rel=1e-9) for value in result.returns)
    assert all(value == pytest.approx(1.0, rel=1e-9) for value in result.gross_exposure)

    assert result.turnover[0] == pytest.approx(1.0, rel=1e-9)
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in result.turnover[1:])


def test_a_run_without_a_cost_model_is_gross() -> None:
    """No cost model means no costs, and net equals gross. Explicit, so neither can pose as the
    other: reporting a gross figure as though it were net is the omission Phase 1.1 exists to fix.
    """
    sessions = make_sessions()
    result = build_engine(sessions).run(
        EqualWeightHold(), sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY
    )
    assert all(value == 0.0 for value in result.costs)
    assert result.returns == pytest.approx(result.gross_returns)


def test_the_cost_model_is_charged_per_leg_at_the_session_s_rates() -> None:
    """Costs must bite, and must equal the cost model's own figure for the same legs.

    Only the first fill turns the book over, so only that session carries a cost: three equal buys
    of a third of the book each, no sells, so no depository charge and no stamp duty rebate.
    """
    sessions = make_sessions()
    panel = make_panel(sessions)
    model = CostModel(load_config().costs)
    engine = BacktestEngine(
        panel, make_calendar(sessions), make_universe([(sessions[0], SYMBOLS)]),
        cost_model=model,
    )
    result = engine.run(
        EqualWeightHold(), sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY
    )

    per_name = INITIAL_EQUITY / len(SYMBOLS)
    expected = len(SYMBOLS) * model.execute(
        per_name, "buy", day_traded_value=1_000_000_000.0, on=sessions[1]
    ).total / INITIAL_EQUITY

    assert result.costs[0] == pytest.approx(expected, rel=1e-9)
    assert result.costs[0] > 0.0, "a costed run must differ from a gross one"
    assert all(value == pytest.approx(0.0, abs=1e-15) for value in result.costs[1:])
    # Gross is recorded untouched; net is gross less the cost of getting invested.
    assert result.gross_returns[0] == pytest.approx(DAILY_GAIN, rel=1e-9)
    assert result.returns[0] == pytest.approx(DAILY_GAIN - expected, rel=1e-9)


def test_an_oversized_order_is_recorded_as_a_participation_breach() -> None:
    """The charter demands the limit be enforced, not assumed. Breaches are recorded per order so
    a whole corpus still runs and the frequency of breaches is itself reportable."""
    sessions = make_sessions()
    panel = make_panel(sessions).with_columns(pl.lit(1_000_000.0).alias("turnover_inr"))
    engine = BacktestEngine(
        panel, make_calendar(sessions), make_universe([(sessions[0], SYMBOLS)]),
        cost_model=CostModel(load_config().costs),
    )
    result = engine.run(
        EqualWeightHold(), sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY
    )
    # A third of a million rupees into a session that traded ten lakh is 33%, far past the 1% cap.
    assert len(result.participation_breaches) == len(SYMBOLS)
    assert all(rate > 0.3 for _, _, rate in result.participation_breaches)


def test_a_range_with_fewer_than_two_sessions_is_rejected() -> None:
    """One session cannot carry both a decision and a fill, so it is an error, not an empty run."""
    sessions = make_sessions()
    engine = build_engine(sessions)
    with pytest.raises(ValueError, match="at least two sessions"):
        engine.run(CashOnly(), sessions[0], sessions[0])


# --- the point-in-time guarantee ------------------------------------------------


def test_signal_stamped_with_the_fill_session_raises() -> None:
    """The engine's core promise. A stamp equal to the fill date is lookahead and must raise.

    ``PointInTimeError``, not a warning and not a dropped order: lookahead has to surface as a
    crash, because as a silently better backtest it is invisible.
    """
    sessions = make_sessions()
    engine = build_engine(sessions)
    with pytest.raises(PointInTimeError, match="strictly earlier"):
        engine.run(ClaimsFillSession(), sessions[0], sessions[-1])


def test_signal_stamped_after_the_fill_session_raises() -> None:
    sessions = make_sessions()
    engine = build_engine(sessions)
    with pytest.raises(PointInTimeError, match="strictly earlier"):
        engine.run(ClaimsFuture(), sessions[0], sessions[-1])


def test_signal_stamped_between_the_decision_session_and_the_fill_raises() -> None:
    """A Sunday stamp on a Monday fill is before the fill but after Friday's decision.

    This is the check that separates "earlier than the fill" from "no later than the last session
    the strategy could actually have observed". Only the second is point-in-time discipline.
    """
    sessions = make_sessions()
    engine = build_engine(sessions)
    with pytest.raises(PointInTimeError, match="later than the decision session"):
        engine.run(ClaimsCalendarDayBeforeFill(), sessions[0], sessions[-1])


def test_market_view_cannot_see_the_decision_date_or_later() -> None:
    """Constructed directly, the view holds nothing stamped on or after ``as_of``."""
    sessions = make_sessions()
    view = MarketView(make_panel(sessions), as_of=sessions[10], symbols=SYMBOLS)

    history = view.history()
    assert history.height > 0
    assert history.filter(pl.col("session_date") >= view.as_of).height == 0
    assert history["session_date"].max() == sessions[9]

    # The lookback window counts sessions, not rows, across a three-symbol panel.
    recent = view.history(lookback=3)
    assert sorted(recent["session_date"].unique().to_list()) == sessions[7:10]
    assert set(view.latest_close()) == set(SYMBOLS)


def test_market_view_require_before_rejects_the_decision_date() -> None:
    """The explicit guard a strategy calls when it wants to prove a timestamp is in the past."""
    sessions = make_sessions()
    view = MarketView(make_panel(sessions), as_of=sessions[10], symbols=SYMBOLS)

    view.require_before(sessions[9])  # the last legitimately visible session
    with pytest.raises(PointInTimeError):
        view.require_before(sessions[10])
    with pytest.raises(PointInTimeError):
        view.require_before(sessions[11])


def test_every_view_the_engine_builds_is_truncated_before_its_fill_session() -> None:
    """The same property, checked on the views the engine actually hands out during a run.

    Testing ``MarketView`` in isolation only proves the class behaves; this proves the engine
    constructs it with the right ``as_of`` on every one of the 29 fills.
    """
    sessions = make_sessions()
    engine = build_engine(sessions)
    strategy = RecordingStrategy()
    engine.run(strategy, sessions[0], sessions[-1])

    assert len(strategy.views) == len(sessions) - 1
    for offset, view in enumerate(strategy.views):
        fill_session = sessions[offset + 1]
        assert view.as_of == fill_session
        history = view.history()
        assert history.filter(pl.col("session_date") >= fill_session).height == 0
        # The newest visible session is the decision session, one step back from the fill.
        assert history["session_date"].max() == sessions[offset]


# --- exposure and universe controls ---------------------------------------------


def test_gross_exposure_is_clamped_to_the_configured_cap() -> None:
    """A strategy asking for 5x is scaled back to the cap, not rejected and not obeyed.

    The cap is set to 0.8 rather than the default 1.0 so the test would fail if the engine were
    clamping to a hard-coded value instead of reading its configuration. Relative weights survive
    the scaling: the 2:2:1 book stays 2:2:1.
    """
    sessions = make_sessions()
    engine = build_engine(sessions, max_gross_exposure=0.8)
    strategy = FixedWeights({"AAA": 2.0, "BBB": 2.0, "CCC": 1.0})
    result = engine.run(strategy, sessions[0], sessions[-1], initial_equity=INITIAL_EQUITY)

    assert all(value == pytest.approx(0.8, rel=1e-12) for value in result.gross_exposure)
    # Fully invested at 0.8 gross on a panel gaining 1% intraday: 0.8% a session.
    assert all(
        value == pytest.approx(0.8 * DAILY_GAIN, rel=1e-9) for value in result.returns
    )


def test_a_book_inside_the_cap_is_left_alone() -> None:
    """Clamping must be conditional; an under-invested book is not scaled up to the cap."""
    sessions = make_sessions()
    engine = build_engine(sessions, max_gross_exposure=1.0)
    strategy = FixedWeights({"AAA": 0.2, "BBB": 0.1})
    result = engine.run(strategy, sessions[0], sessions[-1])

    assert all(value == pytest.approx(0.3, rel=1e-12) for value in result.gross_exposure)


def test_symbols_outside_the_point_in_time_universe_are_dropped() -> None:
    """Constituency is read as of the decision session, so a name dropped at a rebalance goes.

    The universe holds all three symbols from the start and only AAA and BBB from session 14. A
    strategy that keeps asking for all three should show gross exposure of 0.75 while CCC is a
    constituent and 0.50 afterwards. Getting this wrong is survivorship bias with extra steps.
    """
    sessions = make_sessions()
    universe = make_universe(
        [
            (sessions[0], SYMBOLS),
            (sessions[14], ("AAA", "BBB")),
        ]
    )
    engine = build_engine(sessions, universe=universe)
    strategy = FixedWeights({"AAA": 0.25, "BBB": 0.25, "CCC": 0.25})
    result = engine.run(strategy, sessions[0], sessions[-1])

    exposure_by_date = dict(zip(result.dates, result.gross_exposure, strict=True))
    # Fill at session 14 was decided at session 13, before the rebalance: CCC still in.
    assert exposure_by_date[sessions[14]] == pytest.approx(0.75, rel=1e-12)
    # Fill at session 15 was decided at session 14, the rebalance itself: CCC is out.
    assert exposure_by_date[sessions[15]] == pytest.approx(0.50, rel=1e-12)
    assert exposure_by_date[sessions[-1]] == pytest.approx(0.50, rel=1e-12)


def test_unknown_symbols_are_dropped_rather_than_priced() -> None:
    """A name that was never in the universe contributes nothing, silently or otherwise."""
    sessions = make_sessions()
    engine = build_engine(sessions)
    strategy = FixedWeights({"AAA": 0.5, "ZZZ": 0.5})
    result = engine.run(strategy, sessions[0], sessions[-1])

    assert all(value == pytest.approx(0.5, rel=1e-12) for value in result.gross_exposure)
    assert all(
        value == pytest.approx(0.5 * DAILY_GAIN, rel=1e-9) for value in result.returns
    )


# --- known-artifact flagging ----------------------------------------------------


def test_sessions_holding_a_flagged_name_are_recorded() -> None:
    """Holding a name inside a known-artifacts window marks the session, without altering the run.

    The marker is not an error. It exists so that an apparent edge resting on suspect price data
    can be recognised as such at review time rather than trusted at face value.
    """
    sessions = make_sessions()
    register = make_register([("AAA", sessions[5], sessions[7])])
    engine = build_engine(sessions, artifact_register=register)
    strategy = FixedWeights({"AAA": 1.0})
    result = engine.run(strategy, sessions[0], sessions[-1])

    flagged = set(result.flagged_sessions)
    assert flagged == {sessions[5], sessions[6], sessions[7]}
    assert sessions[4] not in flagged
    assert sessions[8] not in flagged
    # Flagging is a marker only: the equity curve is unchanged by it.
    assert result.equity[-1] == pytest.approx(
        INITIAL_EQUITY * (1.0 + DAILY_GAIN) ** (len(sessions) - 1), rel=1e-9
    )


def test_a_flag_on_a_name_that_is_not_held_does_not_mark_the_session() -> None:
    """The register is checked against holdings, not against the whole panel."""
    sessions = make_sessions()
    register = make_register([("CCC", sessions[5], sessions[7])])
    engine = build_engine(sessions, artifact_register=register)
    strategy = FixedWeights({"AAA": 1.0})
    result = engine.run(strategy, sessions[0], sessions[-1])

    assert result.flagged_sessions == []


def test_no_register_means_no_flagged_sessions() -> None:
    sessions = make_sessions()
    engine = build_engine(sessions)
    result = engine.run(FixedWeights({"AAA": 1.0}), sessions[0], sessions[-1])
    assert result.flagged_sessions == []


# --- result shape ---------------------------------------------------------------


def test_result_frame_has_one_row_per_recorded_session() -> None:
    """The exported frame is what reporting reads, so its shape is pinned here."""
    sessions = make_sessions()
    engine = build_engine(sessions)
    result = engine.run(EqualWeightHold(), sessions[0], sessions[-1])

    frame = result.to_frame()
    assert frame.height == len(sessions) - 1
    assert frame.columns == [
        "session_date",
        "equity",
        "return",
        "gross_exposure",
        "turnover",
        "cost",
        "gross_return",
    ]
    assert frame["session_date"].to_list() == sessions[1:]


# --- position recording (Phase 2.2 instrumentation) -----------------------------


def test_positions_are_not_recorded_unless_asked() -> None:
    """The default must stay exactly as Phase 1.2 was approved: no positions, no extra memory."""
    sessions = make_sessions()
    result = build_engine(sessions).run(EqualWeightHold(), sessions[0], sessions[-1])
    assert result.positions == []


def test_recording_positions_changes_no_number_the_engine_reports() -> None:
    """Instrumentation must be inert. If turning it on moves the equity curve, it is not
    instrumentation — it is a change to the model, and Phase 1.2's approval would no longer cover
    the results. Asserted with ``==`` on the whole series rather than a tolerance.
    """
    sessions = make_sessions()
    plain = build_engine(sessions).run(EqualWeightHold(), sessions[0], sessions[-1])
    instrumented = build_engine(sessions, record_positions=True).run(
        EqualWeightHold(), sessions[0], sessions[-1]
    )

    assert instrumented.dates == plain.dates
    assert instrumented.equity == plain.equity
    assert instrumented.returns == plain.returns
    assert instrumented.gross_returns == plain.gross_returns
    assert instrumented.turnover == plain.turnover
    assert instrumented.costs == plain.costs
    assert instrumented.gross_exposure == plain.gross_exposure


def test_a_recorded_book_matches_the_exposure_reported_for_the_same_session() -> None:
    """The recorded weights are the ones the engine actually traded, not a parallel bookkeeping.

    Checked against ``gross_exposure``, which is computed from the same target inside the loop: if
    the recorded book were the previous session's, or a copy taken at the wrong point, the two
    would disagree on the first session the book changes.
    """
    sessions = make_sessions()
    result = build_engine(sessions, record_positions=True).run(
        EqualWeightHold(), sessions[0], sessions[-1]
    )

    assert len(result.positions) == len(result.dates)
    for book, exposure in zip(result.positions, result.gross_exposure, strict=True):
        assert sum(abs(weight) for weight in book.values()) == pytest.approx(exposure, rel=1e-12)


def test_a_cash_only_book_is_recorded_as_empty_rather_than_omitted() -> None:
    """Holding nothing is a position. Skipping the append would misalign every later session."""
    sessions = make_sessions()
    result = build_engine(sessions, record_positions=True).run(
        CashOnly(), sessions[0], sessions[-1]
    )
    assert len(result.positions) == len(result.dates)
    assert all(book == {} for book in result.positions)
