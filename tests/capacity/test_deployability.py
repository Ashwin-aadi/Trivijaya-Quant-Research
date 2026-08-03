"""Alpha decay curves and constraint-based deployment capacity.

The capacity tests fix arithmetic against hand-computed values, because the whole defence of this
number is that it is arithmetic on observables — if it cannot be checked by hand it has lost the
property that made it worth reporting instead of an impact model.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.capacity.decay import DecayPoint, decay_curve, forward_returns, half_life
from src.capacity.deployability import (
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.common.exceptions import DataIntegrityError

DAY = date(2020, 1, 1)


# --- decay -----------------------------------------------------------------------------------


def test_forward_return_skips_a_session_so_it_cannot_trade_on_the_close_it_saw() -> None:
    """Signal at a close fills at the next open at the earliest; the gap enforces it."""
    closes = [100.0, 110.0, 121.0, 133.1]
    panel = pl.DataFrame(
        {
            "session_date": [DAY + timedelta(days=i) for i in range(4)],
            "symbol": ["AAA"] * 4,
            "adj_close": closes,
        }
    )
    forward = forward_returns(panel, (1,))
    # At session 0 the return runs from close[1] to close[2] -- session 1's own move is skipped.
    assert forward["fwd_1"][0] == pytest.approx(121.0 / 110.0 - 1.0)


def test_overlap_correction_counts_monthly_books_held_a_week_as_independent() -> None:
    """Books formed 21 sessions apart and held 5 do not overlap, so nothing should be discarded."""
    rng = np.random.default_rng(0)
    n = 60
    weights = pl.DataFrame(
        {
            "session_date": [DAY + timedelta(days=21 * i) for i in range(n)],
            "factor": ["f"] * n,
            "symbol": ["AAA"] * n,
            "weight": [1.0] * n,
        }
    )
    forward = pl.DataFrame(
        {
            "session_date": [DAY + timedelta(days=21 * i) for i in range(n)],
            "symbol": ["AAA"] * n,
            "fwd_5": rng.normal(0, 0.01, size=n),
        }
    )
    spaced = decay_curve(weights, forward, horizons=(5,), formation_spacing=21)[0]
    assert spaced.n_non_overlapping == spaced.n_observations

    # Daily formation over the same horizon does overlap, and must be discounted.
    daily = decay_curve(weights, forward, horizons=(5,), formation_spacing=1)[0]
    assert daily.n_non_overlapping == spaced.n_observations // 5


def test_half_life_is_the_horizon_at_which_the_edge_halves() -> None:
    points = [
        DecayPoint("f", 1, 0.0010, 0.0001, 100, 100),
        DecayPoint("f", 5, 0.0007, 0.0001, 100, 20),
        DecayPoint("f", 10, 0.0004, 0.0001, 100, 10),
    ]
    assert half_life(points) == 10.0


def test_no_half_life_is_reported_for_a_factor_that_never_made_money() -> None:
    """A decaying edge is only meaningful for a signal that had one; the alternative is nonsense."""
    losing = [DecayPoint("f", 1, -0.001, 0.0001, 100, 100),
              DecayPoint("f", 5, -0.002, 0.0001, 100, 20)]
    assert half_life(losing) is None
    assert half_life([]) is None


def test_a_curve_that_never_halves_returns_none_rather_than_the_longest_horizon() -> None:
    flat = [DecayPoint("f", 1, 0.001, 0.0001, 100, 100),
            DecayPoint("f", 63, 0.0009, 0.0001, 100, 2)]
    assert half_life(flat) is None


# --- capacity ---------------------------------------------------------------------------------


def test_turnover_counts_a_name_leaving_the_book_at_its_full_previous_weight() -> None:
    """Dropping departures would understate turnover and so overstate capacity, invisibly."""
    weights = pl.DataFrame(
        {
            "session_date": [DAY, DAY, DAY + timedelta(days=1)],
            "factor": ["f"] * 3,
            "symbol": ["AAA", "BBB", "AAA"],
            "weight": [0.5, 0.5, 1.0],
        }
    )
    traded = turnover_by_session(weights)
    second = traded.filter(pl.col("session_date") == DAY + timedelta(days=1))
    departed = second.filter(pl.col("symbol") == "BBB")
    assert departed.height == 1
    assert departed["traded_fraction"][0] == pytest.approx(0.5)


def test_capacity_is_the_tightest_name_and_matches_a_hand_computation() -> None:
    """kappa*ADV/|dw|, minimised. AAA: .01*1e9/.5 = 2e7. BBB: .01*1e8/.25 = 4e6."""
    traded = pl.DataFrame(
        {
            "session_date": [DAY, DAY],
            "factor": ["f", "f"],
            "symbol": ["AAA", "BBB"],
            "traded_fraction": [0.5, 0.25],
        }
    )
    liquidity = pl.DataFrame(
        {"session_date": [DAY, DAY], "symbol": ["AAA", "BBB"], "adv_inr": [1e9, 1e8]}
    )
    out = session_capacity(traded, liquidity, participation_limit=0.01)
    assert out["capacity_inr"][0] == pytest.approx(4e6)
    assert out["binding_symbol"][0] == "BBB"
    assert out["second_capacity"][0] == pytest.approx(2e7)


def test_a_single_traded_name_has_no_second_constraint_rather_than_raising() -> None:
    traded = pl.DataFrame(
        {"session_date": [DAY], "factor": ["f"], "symbol": ["AAA"], "traded_fraction": [0.5]}
    )
    liquidity = pl.DataFrame({"session_date": [DAY], "symbol": ["AAA"], "adv_inr": [1e9]})
    out = session_capacity(traded, liquidity, participation_limit=0.01)
    assert out["second_capacity"][0] is None


def test_a_non_positive_participation_limit_raises() -> None:
    traded = pl.DataFrame(
        {"session_date": [DAY], "factor": ["f"], "symbol": ["A"], "traded_fraction": [0.5]}
    )
    liquidity = pl.DataFrame({"session_date": [DAY], "symbol": ["A"], "adv_inr": [1e9]})
    with pytest.raises(DataIntegrityError):
        session_capacity(traded, liquidity, participation_limit=0.0)


def test_summary_flags_a_session_whose_capacity_rests_on_one_position() -> None:
    per_session = pl.DataFrame(
        {
            "factor": ["f", "f"],
            "session_date": [DAY, DAY + timedelta(days=1)],
            "capacity_inr": [1e6, 1e7],
            "binding_symbol": ["AAA", "BBB"],
            "n_names": [10, 10],
            # First session: next-tightest is 10x looser, so one position sets capacity.
            "second_capacity": [1e7, 1.1e7],
        }
    )
    summary = summarise_capacity(per_session, participation_limit=0.01)[0]
    assert summary.fraction_bound_by_one_name == pytest.approx(0.5)
    assert summary.median_capacity_inr == pytest.approx(5.5e6)
    assert summary.n_rebalance_sessions == 2


def test_relaxed_capacity_reports_what_dropping_the_tightest_name_would_buy() -> None:
    """A robustness check, never a replacement: it says whether the number is about one holding."""
    per_session = pl.DataFrame(
        {
            "factor": ["f", "f"],
            "session_date": [DAY, DAY + timedelta(days=1)],
            "capacity_inr": [1e6, 1e7],
            "binding_symbol": ["AAA", "BBB"],
            "n_names": [10, 10],
            "second_capacity": [1e7, 1.1e7],
        }
    )
    summary = summarise_capacity(per_session, participation_limit=0.01)[0]
    assert summary.median_relaxed_capacity_inr == pytest.approx(1.05e7)
    # Relaxing must never tighten capacity; dropping a constraint can only loosen it.
    assert summary.median_relaxed_capacity_inr >= summary.median_capacity_inr
    assert summary.relaxed_over_constrained == pytest.approx(1.05e7 / 5.5e6)


def test_relaxed_capacity_is_undefined_rather_than_infinite_for_one_name_books() -> None:
    per_session = pl.DataFrame(
        {
            "factor": ["f"],
            "session_date": [DAY],
            "capacity_inr": [1e6],
            "binding_symbol": ["AAA"],
            "n_names": [1],
            "second_capacity": [None],
        },
        schema_overrides={"second_capacity": pl.Float64},
    )
    summary = summarise_capacity(per_session, participation_limit=0.01)[0]
    assert summary.median_relaxed_capacity_inr != summary.median_relaxed_capacity_inr  # NaN


def test_flow_conditional_capacity_carries_the_session_count_for_every_cell() -> None:
    """The ratio between states is what a reader quotes, so it must never appear without its n."""
    per_session = pl.DataFrame(
        {
            "factor": ["f"] * 4,
            "session_date": [DAY + timedelta(days=i) for i in range(4)],
            "capacity_inr": [1e6, 2e6, 5e6, 7e6],
        }
    )
    flows = pl.DataFrame(
        {
            "session_date": [DAY + timedelta(days=i) for i in range(4)],
            "flow_state": ["outflow", "outflow", "inflow", "inflow"],
        }
    )
    out = capacity_by_flow_state(per_session, flows)
    assert set(out["n_sessions"].to_list()) == {2}
    inflow = out.filter(pl.col("flow_state") == "inflow")["median_capacity_inr"][0]
    assert inflow == pytest.approx(6e6)


def test_unlabelled_flow_sessions_are_dropped_not_counted_as_a_state() -> None:
    per_session = pl.DataFrame(
        {"factor": ["f"] * 2, "session_date": [DAY, DAY + timedelta(days=1)],
         "capacity_inr": [1e6, 2e6]}
    )
    flows = pl.DataFrame(
        {"session_date": [DAY, DAY + timedelta(days=1)], "flow_state": [None, "inflow"]}
    )
    out = capacity_by_flow_state(per_session, flows)
    assert out.height == 1
    assert out["n_sessions"][0] == 1
