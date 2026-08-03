"""Impact primitives and the identifiability diagnostics that Phase 3.0's halt condition rests on.

Two of these tests are positive controls rather than unit tests, and they are the load-bearing
ones. The Phase 3.0 report concludes that the exponent is unstable and that heavy-volume price
moves do not reverse. Both conclusions are drawn from estimators written for this phase, so a
reader is entitled to ask whether the estimators can recover *anything*. These tests build series
with a known exponent and a known reversal and confirm they are found.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.capacity.impact import (
    add_daily_measures,
    amihud_illiquidity,
    extrapolation_gap,
    fit_elasticity,
    participation_rate,
    reversal_betas,
    square_root_impact,
)
from src.common.exceptions import DataIntegrityError


def _panel(closes: list[float], turnover: list[float], symbol: str = "AAA") -> pl.DataFrame:
    start = date(2020, 1, 1)
    return pl.DataFrame(
        {
            "session_date": [start + timedelta(days=i) for i in range(len(closes))],
            "symbol": [symbol] * len(closes),
            "adj_close": closes,
            "turnover_inr": turnover,
            "volume": [t / max(c, 1e-9) for t, c in zip(turnover, closes, strict=True)],
        }
    )


# --- measures -------------------------------------------------------------------------------


def test_participation_against_a_dead_session_raises_rather_than_returning_infinity() -> None:
    """An order on a session with no trading was not executable, not expensively executable."""
    with pytest.raises(DataIntegrityError):
        participation_rate(1_000_000.0, 0.0)


def test_participation_is_the_plain_ratio() -> None:
    assert participation_rate(1_000.0, 100_000.0) == pytest.approx(0.01)


def test_square_root_impact_is_concave_in_size() -> None:
    """Doubling the order must less than double the impact; that is the whole point of the form."""
    small = square_root_impact(0.01, 0.02, 1.0)
    large = square_root_impact(0.02, 0.02, 1.0)
    assert large < 2 * small
    assert large == pytest.approx(small * math.sqrt(2))


def test_square_root_impact_matches_the_formula_it_cites() -> None:
    assert square_root_impact(0.04, 0.02, 1.5) == pytest.approx(1.5 * 0.02 * 0.2)


@pytest.mark.parametrize(("participation", "volatility"), [(-0.01, 0.02), (0.01, -0.02)])
def test_square_root_impact_rejects_negative_inputs(participation: float,
                                                    volatility: float) -> None:
    with pytest.raises(DataIntegrityError):
        square_root_impact(participation, volatility, 1.0)


def test_amihud_is_the_mean_absolute_return_per_rupee() -> None:
    returns = pl.Series([0.02, -0.04])
    value = pl.Series([1_000_000.0, 2_000_000.0])
    assert amihud_illiquidity(returns, value) == pytest.approx((0.02 / 1e6 + 0.04 / 2e6) / 2)


def test_amihud_drops_zero_volume_sessions_rather_than_calling_them_infinitely_illiquid() -> None:
    returns = pl.Series([0.02, 0.05])
    value = pl.Series([1_000_000.0, 0.0])
    assert amihud_illiquidity(returns, value) == pytest.approx(0.02 / 1e6)


def test_amihud_with_no_traded_session_raises() -> None:
    with pytest.raises(DataIntegrityError):
        amihud_illiquidity(pl.Series([0.01]), pl.Series([0.0]))


# --- the point-in-time property -------------------------------------------------------------


def test_trailing_adv_never_includes_the_row_it_normalises() -> None:
    """The defect this guards is invisible in output: a ratio partly a function of itself.

    Session 21 carries a turnover spike. If ADV on that session included it, the spike would
    partly normalise itself away and the relative traded value would be understated. The expected
    value below is computed from the twenty-one sessions *before* it, all of which are 100.
    """
    turnover = [100.0] * 21 + [10_000.0] + [100.0] * 5
    closes = [10.0] * len(turnover)
    measured = add_daily_measures(_panel(closes, turnover), adv_window=21)
    spike = measured.filter(pl.col("turnover_inr") == 10_000.0)
    assert spike.height == 1
    assert spike["adv_inr"][0] == pytest.approx(100.0)
    assert spike["rel_value"][0] == pytest.approx(100.0)


def test_the_first_sessions_have_no_adv_rather_than_a_short_window_one() -> None:
    measured = add_daily_measures(_panel([10.0] * 30, [100.0] * 30), adv_window=21)
    assert measured["adv_inr"][:21].null_count() == 21
    assert measured["adv_inr"][21] is not None


def test_a_panel_missing_a_column_raises_at_the_boundary() -> None:
    with pytest.raises(DataIntegrityError):
        add_daily_measures(pl.DataFrame({"session_date": [date(2020, 1, 1)], "symbol": ["A"]}))


# --- positive controls ----------------------------------------------------------------------


def test_the_exponent_estimator_recovers_an_exponent_it_was_given() -> None:
    """Build |return| = 0.01 * rel_value^0.5 exactly, and require delta back to three places."""
    rng = np.random.default_rng(42)
    n = 800
    rel = rng.uniform(0.2, 5.0, size=n)
    frame = pl.DataFrame(
        {
            "symbol": ["AAA"] * n,
            "session_date": [date(2020, 1, 1) + timedelta(days=i) for i in range(n)],
            "abs_ret": 0.01 * rel**0.5,
            "rel_value": rel,
        }
    )
    fits = fit_elasticity(frame, min_sessions=100)
    assert len(fits) == 1
    assert fits[0].delta == pytest.approx(0.5, abs=1e-3)
    assert fits[0].r_squared == pytest.approx(1.0, abs=1e-6)


def test_the_reversal_estimator_recovers_a_reversal_that_is_there() -> None:
    """A series built to give back half of each move must be measured as beta near -0.5.

    This is the control for the report's central negative finding. The real data returns a beta of
    about +0.02 at this horizon; if this test failed, that +0.02 would mean nothing.
    """
    rng = np.random.default_rng(7)
    n = 1200
    shocks = rng.normal(0, 0.02, size=n)
    closes = [100.0]
    for i in range(1, n):
        # Today's move is this session's shock less half of the previous session's — an exact,
        # known one-session reversal of one half.
        step = shocks[i] - 0.5 * shocks[i - 1]
        closes.append(closes[-1] * (1 + step))
    frame = _panel(closes, [1_000_000.0] * n)
    measured = add_daily_measures(frame, adv_window=21)
    fits = reversal_betas(measured, horizon=1, min_sessions=100)
    assert len(fits) == 1
    assert fits[0].beta < -0.3


def test_reversal_restriction_to_heavy_sessions_reduces_the_sample_it_fits() -> None:
    rng = np.random.default_rng(3)
    n = 900
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.01, size=n)))
    turnover = list(rng.lognormal(15, 1.0, size=n))
    measured = add_daily_measures(_panel(closes, turnover), adv_window=21)
    full = reversal_betas(measured, horizon=5, min_sessions=50)
    heavy = reversal_betas(measured, horizon=5, min_sessions=50, high_participation_quantile=0.9)
    assert heavy[0].n_sessions < full[0].n_sessions


# --- the extrapolation gap ------------------------------------------------------------------


def test_extrapolation_gap_reports_the_distance_in_orders_of_magnitude() -> None:
    frame = pl.DataFrame({"rel_value": [1.0] * 100})
    gap = extrapolation_gap(frame, target_participation=0.01)
    assert gap.orders_of_magnitude == pytest.approx(2.0)
    assert gap.fraction_below_target == 0.0
    assert gap.n_symbol_days == 100


def test_extrapolation_gap_counts_the_sessions_inside_the_region_of_interest() -> None:
    frame = pl.DataFrame({"rel_value": [0.001] * 25 + [1.0] * 75})
    gap = extrapolation_gap(frame, target_participation=0.01)
    assert gap.fraction_below_target == pytest.approx(0.25)


# --- the detectability bound, which is what makes the null a finding ------------------------


def test_a_precise_estimate_can_rule_out_smaller_effects_than_a_noisy_one() -> None:
    """The minimum detectable effect must fall as the standard errors do."""
    from src.capacity.impact import ReversalFit, minimum_detectable_beta

    noisy = [ReversalFit(f"S{i}", 0.0, 0.10, 0.0, 200) for i in range(20)]
    precise = [ReversalFit(f"S{i}", 0.0, 0.01, 0.0, 200) for i in range(20)]
    assert (
        minimum_detectable_beta(precise).median_minimum_detectable_beta
        < minimum_detectable_beta(noisy).median_minimum_detectable_beta
    )


def test_pooling_is_sharper_than_any_single_symbol() -> None:
    from src.capacity.impact import ReversalFit, minimum_detectable_beta

    fits = [ReversalFit(f"S{i}", 0.02, 0.05, 0.0, 200) for i in range(100)]
    bound = minimum_detectable_beta(fits)
    assert bound.pooled_minimum_detectable_beta < bound.median_minimum_detectable_beta
    assert bound.pooled_beta == pytest.approx(0.02)


def test_a_sign_disagreement_between_weightings_is_reported_not_hidden() -> None:
    """When precision weighting and equal weighting disagree, that disagreement is the finding."""
    from src.capacity.impact import ReversalFit, minimum_detectable_beta

    # One very precisely estimated negative symbol against many noisy positive ones.
    fits = [ReversalFit("PRECISE", -1.0, 0.001, 0.0, 200)]
    fits += [ReversalFit(f"S{i}", 0.5, 1.0, 0.0, 200) for i in range(50)]
    bound = minimum_detectable_beta(fits)
    assert bound.pooled_beta < 0 < bound.unweighted_mean_beta
    assert bound.estimates_disagree


def test_an_unconventional_power_is_refused_rather_than_approximated() -> None:
    from src.capacity.impact import ReversalFit, minimum_detectable_beta

    fits = [ReversalFit("A", 0.0, 0.01, 0.0, 200)]
    with pytest.raises(DataIntegrityError):
        minimum_detectable_beta(fits, power=0.9)
