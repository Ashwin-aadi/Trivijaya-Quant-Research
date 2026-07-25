"""Tests for the performance statistics in ``src.eval.metrics``.

These are the numbers every downstream claim in the lab is expressed in, so they are pinned
against hand-computable answers rather than against whatever the code happens to return today.
Three properties get particular attention:

* the degenerate inputs — empty, single-observation, and zero-volatility series — because a
  metrics module that raises or divides by zero on those will do it in the middle of a 300-strategy
  sweep, and a metrics module that returns an absurd number instead is worse;
* the sign and bound conventions (drawdown negative-or-zero, tracking error zero against self),
  because these are the conventions the rest of the code reads without re-checking;
* ``summarise`` carrying ``n_sessions``, because a Sharpe ratio without its sample size is not a
  reportable figure in this project.
"""

import math

import pytest

from src.eval.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
    summarise,
    tracking_error,
)

# --- degenerate inputs ----------------------------------------------------------


def test_empty_series_returns_zero_everywhere() -> None:
    """No observations means no claim, and the correct representation of no claim is 0.0."""
    empty: list[float] = []
    assert annualised_return(empty) == 0.0
    assert annualised_volatility(empty) == 0.0
    assert sharpe_ratio(empty) == 0.0
    assert max_drawdown(empty) == 0.0
    assert tracking_error(empty, empty) == 0.0


def test_all_zero_returns_are_exactly_flat() -> None:
    """A cash-only strategy's return series. Every figure must be exactly zero, not nearly zero.

    Exact equality is deliberate. This is the series the engine produces for a strategy that holds
    nothing, and any drift here would show up as a fictitious edge or a fictitious loss.
    """
    flat = [0.0] * 250
    assert annualised_return(flat) == 0.0
    assert annualised_volatility(flat) == 0.0
    assert sharpe_ratio(flat) == 0.0
    assert max_drawdown(flat) == 0.0


def test_constant_return_series_has_no_volatility_and_no_sharpe() -> None:
    """Zero dispersion must not become a division by zero.

    The constant is a power of two on purpose. ``sharpe_ratio`` guards with an exact
    ``volatility == 0`` comparison, so the guard is only reachable when the sample variance lands
    on exactly 0.0 in binary floating point. 1/64 does; 0.01 does not, and that gap is pinned
    separately in ``test_a_near_constant_series_slips_past_the_zero_volatility_guard`` below.
    """
    constant = [0.015625] * 64
    assert annualised_volatility(constant) == 0.0
    assert sharpe_ratio(constant) == 0.0
    # The return itself is emphatically not zero — only the risk is.
    assert annualised_return(constant) > 0.0


def test_a_near_constant_series_slips_past_the_zero_volatility_guard() -> None:
    """Pins a known fragility in ``sharpe_ratio`` rather than leaving it undocumented.

    For a mathematically constant series of 0.01 the sample variance is not exactly zero in
    floating point — roughly 1e-16 of residue survives — so the ``volatility == 0`` guard does not
    fire and the ratio explodes to order 1e16. This is asserted, not fixed, because the fix is a
    tolerance the PI has to choose. Raised in the phase report; do not silently change either the
    source or this test.
    """
    near_constant = [0.01] * 250
    volatility = annualised_volatility(near_constant)
    assert volatility != 0.0            # mathematically it is zero
    assert volatility < 1e-12           # numerically it is float residue, nothing more
    assert sharpe_ratio(near_constant) > 1e10


def test_single_observation_has_no_dispersion_and_no_sharpe() -> None:
    """One session cannot support a volatility or a Sharpe, so both are 0.0 rather than an error.

    ``annualised_return`` is not similarly guarded: it happily compounds a single session across a
    whole year and produces an absurd figure. That is not a defect so much as the reason every
    number in this project is reported next to its sample size.
    """
    assert annualised_volatility([0.01]) == 0.0
    assert sharpe_ratio([0.01]) == 0.0
    assert annualised_return([0.0]) == 0.0
    # One good day annualised over 250 sessions: 1.01**250 - 1, comfortably above 1000%.
    assert annualised_return([0.01]) > 10.0


def test_total_loss_returns_minus_one_rather_than_a_complex_root() -> None:
    """Equity reaching zero cannot be raised to a fractional power, so the floor is -100%."""
    assert annualised_return([-1.0]) == -1.0
    assert annualised_return([-1.0, 0.5, 0.5]) == -1.0


# --- known answers --------------------------------------------------------------


def test_annualised_return_known_answer_doubling_over_two_years() -> None:
    """Doubling over 500 sessions (two trading years) annualises to sqrt(2) - 1, about 41.4%."""
    per_session = 2.0 ** (1.0 / 500) - 1.0
    returns = [per_session] * 500
    assert annualised_return(returns) == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-9)


def test_annualised_volatility_known_answer() -> None:
    """Two observations of +1% and -1%: sample sd is 0.01*sqrt(2), annualised by sqrt(250)."""
    expected = 0.01 * math.sqrt(2.0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert annualised_volatility([0.01, -0.01]) == pytest.approx(expected, rel=1e-12)


def test_max_drawdown_on_a_hand_built_series() -> None:
    """+10%, then -50%, then +10%.

    Equity runs 1.00 -> 1.10 -> 0.55 -> 0.605 against a running peak of 1.10. The worst point is
    0.55/1.10 - 1 = -50%; the recovery to 0.605 leaves the trough unchanged at -50%, because a
    drawdown is measured to the trough and not to the end of the series.
    """
    assert max_drawdown([0.10, -0.50, 0.10]) == pytest.approx(-0.50, rel=1e-12)


def test_max_drawdown_is_never_positive() -> None:
    """A monotonically rising series has never fallen from a peak, so its drawdown is exactly 0."""
    assert max_drawdown([0.01] * 10) == 0.0

    mixed = [0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.05]
    assert max_drawdown(mixed) <= 0.0


def test_sharpe_ratio_subtracts_the_risk_free_rate() -> None:
    """The risk-free argument must actually move the number; it is not decoration.

    An Indian backtest run against a ~6-7% policy rate is a materially different claim from one
    run against zero, so a silently ignored argument would be a real misstatement.
    """
    returns = [0.01, -0.005, 0.012, -0.004, 0.008, 0.002, -0.001, 0.006]
    gross = sharpe_ratio(returns, risk_free_rate=0.0)
    net = sharpe_ratio(returns, risk_free_rate=0.06)
    volatility = annualised_volatility(returns)
    assert net == pytest.approx(gross - 0.06 / volatility, rel=1e-12)


# --- tracking error -------------------------------------------------------------


def test_tracking_error_against_itself_is_exactly_zero() -> None:
    """Every difference is 0.0 exactly, so no floating-point slack is needed here."""
    returns = [0.01, -0.02, 0.003, 0.0, -0.011, 0.007]
    assert tracking_error(returns, returns) == 0.0


def test_tracking_error_against_a_flat_benchmark_is_own_volatility() -> None:
    """With a zero benchmark the difference series is the return series itself."""
    returns = [0.01, -0.02, 0.003, 0.0, -0.011, 0.007]
    flat = [0.0] * len(returns)
    assert tracking_error(returns, flat) == pytest.approx(annualised_volatility(returns), rel=1e-12)


def test_tracking_error_rejects_a_length_mismatch() -> None:
    """Silently truncating to the shorter series would compare two different periods."""
    with pytest.raises(ValueError, match="lengths differ"):
        tracking_error([0.01, 0.02, 0.03], [0.01, 0.02])


# --- summarise ------------------------------------------------------------------


def test_summarise_reports_the_sample_size_alongside_every_figure() -> None:
    """No figure leaves this module without ``n_sessions`` attached.

    A Sharpe from forty observations and a Sharpe from a thousand are different claims, and the
    number alone does not say which it is. If this key ever disappears, every report built on
    ``summarise`` becomes unciteable.
    """
    returns = [0.01, -0.02, 0.003, 0.0, -0.011, 0.007, 0.002]
    summary = summarise(returns)

    assert summary["n_sessions"] == float(len(returns))
    assert set(summary) == {
        "n_sessions",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "max_drawdown",
    }


def test_summarise_matches_the_individual_functions() -> None:
    """The convenience wrapper must not drift from the primitives it wraps."""
    returns = [0.01, -0.02, 0.003, 0.0, -0.011, 0.007, 0.002]
    summary = summarise(returns)

    assert summary["annualised_return"] == annualised_return(returns)
    assert summary["annualised_volatility"] == annualised_volatility(returns)
    assert summary["sharpe_ratio"] == sharpe_ratio(returns)
    assert summary["max_drawdown"] == max_drawdown(returns)


def test_summarise_of_an_empty_series_is_zeros_with_a_zero_sample_size() -> None:
    """An empty run must be reportable as "no observations", not crash the reporting layer."""
    summary = summarise([])
    assert summary["n_sessions"] == 0.0
    assert all(value == 0.0 for value in summary.values())
