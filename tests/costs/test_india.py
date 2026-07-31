"""Hand-computed verification of the Indian cost model.

The charter requires one ₹1,00,000 delivery round trip worked out line by line by hand, with the
code asserted to match to the paisa. That is the test below, and the arithmetic is written out in
full so a reader can check it without running anything.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.common.config import load_config
from src.common.exceptions import ConfigError
from src.costs.india import CostModel

# --------------------------------------------------------------------------------------------
# HAND COMPUTATION — ₹1,00,000 equity delivery round trip, NSE, discount broker
#
# INDEPENDENTLY VERIFIED BY THE PI on 2026-07-31 against Zerodha's public brokerage calculator
# (equity delivery, NSE, buy 1000 @ 100, sell 1000 @ 100, turnover ₹2,00,000):
#
#     brokerage 0.00   STT 200.00   exchange charges 6.14   GST 1.14
#     SEBI 0.20        stamp duty 15.00                     TOTAL ₹222.48
#
# which is this model's statutory total to the paisa. The ₹6.14 exchange line also confirms the
# exchange/IPFT split: (2.97 + 0.10) x 2 = 6.14, i.e. Zerodha's bundled "0.00307%".
# Zerodha's calculator does NOT include the depository charge, so the DP figure sits on top.
#
# Rates (config.yaml, sourced 2026-07-31, current epoch of the time-varying schedule):
#   STT delivery            0.1%      both sides
#   NSE transaction charge  0.00297%  both sides   (from 2024-10-01)
#   NSE IPFT                0.0001%   both sides   (from 2023-04-01)
#   SEBI turnover fee       0.0001%   both sides   (₹10 per crore)
#   Stamp duty              0.015%    BUY ONLY     (uniform regime, from 2020-07-01)
#   GST                     18%       on (brokerage + exchange + IPFT + SEBI)
#   Brokerage               ₹0                     (zero-brokerage delivery)
#   DP charge               ₹15.34    per scrip, SELL ONLY   (retail mode, the default)
#                           ₹3.50     per scrip, SELL ONLY   (research mode, CDSL component)
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
#   DP charge         = 0 (no scrip charged)                         =   0.0000
#                                                                      ---------
#   SELL TOTAL, statutory                                            = 103.7406
#
#   ROUND TRIP, STATUTORY = 118.7406 + 103.7406                      = 222.4812  <-- PI VERIFIED
#   as a fraction of the ₹1,00,000 notional                          =   0.2225%
#
# Plus the per-scrip depository charge on the sell leg, one scrip:
#   retail mode   222.4812 + 15.34                                   = 237.8212  (default)
#   research mode 222.4812 +  3.50                                   = 225.9812
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
    leg = model.charge_leg(NOTIONAL, "sell", n_scrips=0)
    assert leg.stt == pytest.approx(100.0, abs=PAISA)
    assert leg.stamp_duty == pytest.approx(0.0, abs=PAISA), "stamp duty is buy-side only"
    assert leg.statutory_total == pytest.approx(103.7406, abs=PAISA)


def test_round_trip_matches_the_pi_verified_zerodha_figure(model: CostModel) -> None:
    """₹222.48 statutory, checked by the PI against Zerodha's calculator on 2026-07-31.

    This is the external verification the charter requires, pinned as a test so a later edit to a
    rate cannot quietly move a figure a human confirmed by hand.
    """
    buy, sell = model.round_trip(NOTIONAL, n_scrips=0)
    total = buy.statutory_total + sell.statutory_total
    assert total == pytest.approx(222.4812, abs=PAISA)
    # Roughly 22.2 basis points on a round trip. A strategy turning its book over monthly pays this
    # twelve times a year, which is the whole reason this module exists.
    assert total / NOTIONAL == pytest.approx(0.0022248, abs=1e-6)
    # And Zerodha's own exchange-charge line, which bundles IPFT into one 0.00307% figure.
    assert 2 * (buy.exchange_charge + buy.ipft) == pytest.approx(6.14, abs=PAISA)


def test_round_trip_with_the_retail_depository_charge(model: CostModel) -> None:
    """Retail is the default mode: what an investor is actually billed, not the CDSL floor."""
    buy, sell = model.round_trip(NOTIONAL, n_scrips=1)
    assert sell.dp_charge == pytest.approx(15.34, abs=PAISA)
    assert buy.statutory_total + sell.statutory_total == pytest.approx(237.8212, abs=PAISA)


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


# ---------------------------------------------------------------------------------------------
# Time-varying schedule (PI decision, Checkpoint 1.1 question 2). Only one rate actually moves
# over the study window — the NSE cash transaction charge — but it must move, and it must move on
# the right date, or a 2020 fill is silently charged 2026 rates.
# ---------------------------------------------------------------------------------------------


def test_a_2020_trade_is_charged_2020_rates(model: CostModel) -> None:
    old = model.charge_leg(NOTIONAL, "buy", on=date(2020, 6, 1))
    assert old.exchange_charge == pytest.approx(3.25, abs=PAISA), "0.00325%, the pre-2024 rate"


def test_the_nse_charge_steps_down_on_each_sourced_date(model: CostModel) -> None:
    """0.00325% -> 0.00322% on 2024-04-01 -> 0.00297% on 2024-10-01, each independently sourced."""
    def exchange_on(day: date) -> float:
        return model.charge_leg(NOTIONAL, "buy", on=day).exchange_charge

    assert exchange_on(date(2024, 3, 31)) == pytest.approx(3.25, abs=PAISA)
    assert exchange_on(date(2024, 4, 1)) == pytest.approx(3.22, abs=PAISA)
    assert exchange_on(date(2024, 9, 30)) == pytest.approx(3.22, abs=PAISA)
    assert exchange_on(date(2024, 10, 1)) == pytest.approx(2.97, abs=PAISA)


def test_stt_does_not_move_over_the_study_window(model: CostModel) -> None:
    """STT on equity delivery was 0.1% both sides throughout. Budget 2026 moved F&O only."""
    for day in (date(2020, 1, 1), date(2023, 6, 15), date(2025, 12, 31), date(2026, 7, 1)):
        assert model.charge_leg(NOTIONAL, "buy", on=day).stt == pytest.approx(100.0, abs=PAISA)


def test_a_trade_before_the_schedule_begins_raises(model: CostModel) -> None:
    """Better to refuse than to charge rates that were not in force."""
    with pytest.raises(ConfigError, match="no cost schedule covers"):
        model.charge_leg(NOTIONAL, "buy", on=date(2010, 1, 1))


def test_omitting_the_date_prices_at_the_current_epoch(model: CostModel) -> None:
    """The default is for broker-calculator comparison. The engine always passes a date."""
    assert model.charge_leg(NOTIONAL, "buy").exchange_charge == pytest.approx(
        model.charge_leg(NOTIONAL, "buy", on=date(2026, 7, 31)).exchange_charge
    )


def test_dp_modes_differ_and_retail_is_the_default() -> None:
    """Research mode isolates the CDSL component; retail is what an investor actually pays."""
    costs = load_config().costs
    assert costs.dp_mode == "retail"
    assert costs.dp_charge_by_mode["research"] == pytest.approx(3.50)
    assert costs.dp_charge_by_mode["retail"] == pytest.approx(15.34)
    assert costs.dp_charge_per_scrip_sell == pytest.approx(15.34)
