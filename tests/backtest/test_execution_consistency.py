"""Pins the return accrual to the execution rule, so the two cannot drift apart.

The engine states one timing rule: a signal formed at session *t*'s close fills at session *t+1*'s
open. Return accrual has to follow from that same rule rather than approximate it — a position
already owned at last night's close earns close-to-close including the overnight gap, and only the
portion bought this morning earns from the open.

That is easy to state and easy to break later, because the accrual and the fill timing live in
different parts of the loop. If someone changes one and not the other, the fixtures might still
look right while every held position quietly earns the wrong thing. These tests compute the
expected return from the execution rule directly, by hand, and assert the engine agrees.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.strategy import MarketView, Signal, Strategy
from src.data.calendar import TradingCalendar


def weekdays(start: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = start
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


SESSIONS = weekdays(date(2024, 1, 1), 6)
SYMBOL = "AAA"

# An overnight gap on every session: each open is 2% above the prior close, and each close is a
# further 1% above its own open. If the engine ever reverts to pricing held positions open-to-close
# it will earn only the 1% and the tests below will fail loudly.
OPENS = [100.0, 103.0, 106.09, 109.27, 112.55, 115.93]
CLOSES = [101.0, 104.03, 107.15, 110.36, 113.67, 117.09]


def build_panel() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [SYMBOL] * len(SESSIONS),
            "session_date": SESSIONS,
            "adj_open": OPENS,
            "adj_close": CLOSES,
        }
    )


def build_universe() -> pl.DataFrame:
    return pl.DataFrame(
        {"rebalance_date": [SESSIONS[0]], "symbol": [SYMBOL], "rank": [1]}
    )


class AlwaysHold(Strategy):
    """Holds the single symbol at full weight from the first fill onwards."""

    rationale = "Hold one name continuously, to isolate how a carried position accrues return."

    def generate(self, view: MarketView) -> Signal:
        history = view.history()
        if history.is_empty():
            return Signal(information_available_at=date(1900, 1, 1), weights={})
        stamp = history["session_date"].max()
        assert isinstance(stamp, date)
        return Signal(information_available_at=stamp, weights={SYMBOL: 1.0})


def run() -> list[float]:
    calendar = TradingCalendar(SESSIONS)
    engine = BacktestEngine(build_panel(), calendar, build_universe())
    return engine.run(AlwaysHold(), SESSIONS[0], SESSIONS[-1]).returns


def test_first_fill_earns_open_to_close() -> None:
    """The opening position is bought at the fill session's open, so it earns from there.

    It cannot earn the overnight gap into that session: at the previous close the strategy held
    nothing.
    """
    returns = run()
    expected = CLOSES[1] / OPENS[1] - 1.0
    assert returns[0] == pytest.approx(expected)


def test_carried_position_earns_close_to_close_including_the_gap() -> None:
    """Once held, the position earns the full close-to-close move.

    This is the assertion the overnight bug would have failed. Under the old open-to-close
    accrual the engine returned roughly 1% here instead of roughly 3%, because it implicitly sold
    at every close and re-bought at every open.
    """
    returns = run()
    for index in range(1, len(returns)):
        expected = CLOSES[index + 1] / CLOSES[index] - 1.0
        assert returns[index] == pytest.approx(expected), (
            f"session {index} accrued {returns[index]:.6f} but the execution rule implies "
            f"{expected:.6f}; carried positions must earn close-to-close"
        )


def test_overnight_gap_is_not_discarded() -> None:
    """A direct check that the gap is present, independent of the formulas above.

    Each close-to-close move here is about 3% while each open-to-close move is about 1%. A carried
    session accruing near 1% would mean the gap was dropped.
    """
    returns = run()
    carried = returns[1:]
    assert all(r > 0.02 for r in carried), (
        f"carried sessions accrued {carried}, which looks like open-to-close pricing"
    )


def test_compounded_equity_matches_the_execution_rule() -> None:
    """End-to-end: buying at the first fill open and holding to the last close.

    Derived from the execution rule alone, with no reference to how the engine computes anything.
    """
    returns = run()
    compounded = 1.0
    for r in returns:
        compounded *= 1.0 + r
    expected = CLOSES[-1] / OPENS[1]
    assert compounded == pytest.approx(expected)
