"""Hand-computed verification of the Indian cost model.

The charter requires one ₹1,00,000 delivery round trip worked out line by line by hand, with the
code asserted to match to the paisa. That is the test below, and the arithmetic is written out in
full so a reader can check it without running anything.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.common.config import load_config
from src.costs.india import CostModel

# --------------------------------------------------------------------------------------------
# HAND COMPUTATION — ₹1,00,000 equity delivery round trip, NSE, discount broker
#
# Rates (config.yaml, sourced 2026-07-31):
#   STT delivery            0.1%      both sides
#   NSE transaction charge  0.00297%  both sides   (from 2024-10-01)
#   NSE IPFT                0.0001%   both sides   (from 2023-04-01)
#   SEBI turnover fee       0.0001%   both sides   (₹10 per crore)
#   Stamp duty              0.015%    BUY ONLY     (uniform regime, from 2020-07-01)
#   GST                     18%       on (brokerage + exchange + IPFT + SEBI)
#   Brokerage               ₹0                     (zero-brokerage delivery)
#   DP charge               ₹3.50     per scrip, SELL ONLY
#
# BUY LEG, turnover ₹1,00,000
#   brokerage         = 0                                            =   0.0000
#   STT               = 100000 x 0.001                               = 100.0000
#   exchange charge   = 100000 x 0.0000297                           =   2.9700
#   IPFT              = 100000 x 0.000001                            =   0.1000
#   SEBI turnover fee = 100000 x 0.000001                            =   0.1000
#   stamp duty        = 100000 x 0.00015                             =  15.0000
#   GST               = (0 + 2.97 + 0.10 + 0.10) x 0.18
#                     = 3.17 x 0.18                                  =   0.5706
#   DP charge         = 0 (buy side)                                 =   0.0000
#                                                                      ---------
#   BUY TOTAL                                                        = 118.7406
#
# SELL LEG, turnover ₹1,00,000, one scrip
#   brokerage         = 0                                            =   0.0000
#   STT               = 100000 x 0.001                               = 100.0000
#   exchange charge   = 100000 x 0.0000297                           =   2.9700
#   IPFT              = 100000 x 0.000001                            =   0.1000
#   SEBI turnover fee = 100000 x 0.000001                            =   0.1000
#   stamp duty        = 0 (sell side)                                =   0.0000
#   GST               = 3.17 x 0.18                                  =   0.5706
#   DP charge         = 3.50 x 1                                     =   3.5000
#                                                                      ---------
#   SELL TOTAL                                                       = 107.2406
#
#   ROUND TRIP TOTAL  = 118.7406 + 107.2406                          = 225.9812
#   as a fraction of the ₹1,00,000 notional                          =   0.2260%
#
# Statutory only, excluding the ₹3.50 depository charge:
#   118.7406 + 103.7406                                              = 222.4812
# --------------------------------------------------------------------------------------------

NOTIONAL = 100_000.0
PAISA = 0.005  # half a paisa: "matches to the paisa" means agreeing at 2 decimal places


@pytest.fixture(scope="module")
def model() -> CostModel:
    return CostModel(load_config().costs)


def test_buy_leg_matches_hand_computation(model: CostModel) -> None:
    leg = model.charge_leg(NOTIONAL, "buy")
    assert leg.brokerage == pytest.approx(0.0, abs=PAISA)
    assert leg.stt == pytest.approx(100.0, abs=PAISA)
    assert leg.exchange_charge == pytest.approx(2.97, abs=PAISA)
    assert leg.ipft == pytest.approx(0.10, abs=PAISA)
    assert leg.sebi_fee == pytest.approx(0.10, abs=PAISA)
    assert leg.stamp_duty == pytest.approx(15.0, abs=PAISA)
    assert leg.gst == pytest.approx(0.5706, abs=PAISA)
    assert leg.dp_charge == pytest.approx(0.0, abs=PAISA)
    assert leg.statutory_total == pytest.approx(118.7406, abs=PAISA)


def test_sell_leg_matches_hand_computation(model: CostModel) -> None:
    leg = model.charge_leg(NOTIONAL, "sell", n_scrips=1)
    assert leg.stt == pytest.approx(100.0, abs=PAISA)
    assert leg.stamp_duty == pytest.approx(0.0, abs=PAISA), "stamp duty is buy-side only"
    assert leg.dp_charge == pytest.approx(3.50, abs=PAISA)
    assert leg.statutory_total == pytest.approx(107.2406, abs=PAISA)


def test_round_trip_total_matches_hand_computation(model: CostModel) -> None:
    buy, sell = model.round_trip(NOTIONAL, n_scrips=1)
    total = buy.statutory_total + sell.statutory_total
    assert total == pytest.approx(225.9812, abs=PAISA)
    # Roughly 22.6 basis points on a round trip. A strategy turning its book over monthly pays this
    # twelve times a year, which is the whole reason this module exists.
    assert total / NOTIONAL == pytest.approx(0.002260, abs=1e-6)


def test_round_trip_without_depository_charge(model: CostModel) -> None:
    buy, sell = model.round_trip(NOTIONAL, n_scrips=0)
    assert buy.statutory_total + sell.statutory_total == pytest.approx(222.4812, abs=PAISA)


def test_buy_costs_more_than_sell_because_of_stamp_duty(model: CostModel) -> None:
    """The legs are asymmetric. Code that halves a round-trip figure gets both wrong."""
    buy, sell = model.round_trip(NOTIONAL, n_scrips=0)
    assert buy.statutory_total > sell.statutory_total
    assert buy.statutory_total - sell.statutory_total == pytest.approx(15.0, abs=PAISA)


def test_gst_excludes_stt_and_stamp_duty(model: CostModel) -> None:
    """GST applies to service charges, not to taxes. Taxing STT would inflate every cost."""
    leg = model.charge_leg(NOTIONAL, "buy")
    taxable = leg.brokerage + leg.exchange_charge + leg.ipft + leg.sebi_fee
    assert leg.gst == pytest.approx(taxable * 0.18, abs=1e-9)
    assert leg.gst < leg.stt * 0.18, "GST must not have been levied on STT"


def test_intraday_stt_is_sell_side_only(model: CostModel) -> None:
    buy = model.charge_leg(NOTIONAL, "buy", segment="intraday")
    sell = model.charge_leg(NOTIONAL, "sell", segment="intraday")
    assert buy.stt == pytest.approx(0.0, abs=PAISA)
    assert sell.stt == pytest.approx(NOTIONAL * 0.00025, abs=PAISA)


def test_intraday_is_cheaper_than_delivery(model: CostModel) -> None:
    """Intraday carries lower STT and stamp duty but adds brokerage; delivery should still cost more
    on a round trip at this notional, because 0.1% both sides dominates everything else."""
    d_buy, d_sell = model.round_trip(NOTIONAL, segment="delivery")
    i_buy, i_sell = model.round_trip(NOTIONAL, segment="intraday")
    assert (d_buy.statutory_total + d_sell.statutory_total) > \
           (i_buy.statutory_total + i_sell.statutory_total)


def test_zero_turnover_costs_nothing(model: CostModel) -> None:
    leg = model.charge_leg(0.0, "buy")
    assert leg.statutory_total == pytest.approx(0.0, abs=1e-12)


def test_negative_turnover_raises(model: CostModel) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        model.charge_leg(-1.0, "buy")


# ---------------------------------------------------------------------------------------------
# Slippage and impact. These are ASSUMPTIONS, so the tests assert the *shape* of the model, which
# is what is defensible, rather than any particular level, which is not.
# ---------------------------------------------------------------------------------------------


def test_slippage_rises_with_participation(model: CostModel) -> None:
    """A flat basis-point assumption would make these equal. That is the thing being ruled out."""
    small = model.slippage(10_000.0, day_traded_value=100_000_000.0)
    large = model.slippage(1_000_000.0, day_traded_value=100_000_000.0)
    assert large / 1_000_000.0 > small / 10_000.0


def test_slippage_is_capped(model: CostModel) -> None:
    """An order many times the day's volume must not produce an unbounded cost."""
    absurd = model.slippage(1_000_000_000.0, day_traded_value=1_000.0)
    assert absurd / 1_000_000_000.0 == pytest.approx(0.05, abs=1e-9)


def test_slippage_charges_the_cap_when_nothing_traded(model: CostModel) -> None:
    """Zero traded value gives no basis for an estimate; charging zero would reward illiquidity."""
    assert model.slippage(10_000.0, day_traded_value=0.0) == pytest.approx(10_000.0 * 0.05)


def test_impact_follows_the_square_root_law(model: CostModel) -> None:
    """Quadrupling participation should roughly double the per-rupee impact."""
    base = model.impact(1_000.0, day_traded_value=1_000_000.0) / 1_000.0
    quad = model.impact(4_000.0, day_traded_value=1_000_000.0) / 4_000.0
    assert quad == pytest.approx(2.0 * base, rel=1e-9)


def test_impact_is_zero_without_an_order(model: CostModel) -> None:
    assert model.impact(0.0, day_traded_value=1_000_000.0) == 0.0


def test_execute_sums_statutory_slippage_and_impact(model: CostModel) -> None:
    cost = model.execute(NOTIONAL, "buy", day_traded_value=50_000_000.0)
    assert cost.total == pytest.approx(
        cost.leg.statutory_total + cost.slippage + cost.impact, abs=1e-12
    )
    assert cost.total > cost.leg.statutory_total, "frictions must add to the statutory charges"


def test_staleness_warning_fires_on_the_development_window(model: CostModel) -> None:
    """The development window predates the configured rates, and that must be visible.

    This is a real limitation of applying one rate schedule to a 2020-2024 backtest, and the
    checkpoint asks the PI to choose between a time-varying schedule and a stated constant. The
    test asserts the warning exists so the choice cannot be made by accident.
    """
    warning = model.staleness_warning(date(2020, 1, 1), date(2024, 12, 31))
    assert warning is not None
    assert "anachronistic" in warning


def test_no_staleness_warning_inside_the_effective_period(model: CostModel) -> None:
    assert model.staleness_warning(date(2026, 5, 1), date(2026, 6, 30)) is None
