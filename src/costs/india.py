"""Indian equity transaction costs, charged per trade leg.

Most published Indian backtests are fantasy because they ignore this. A strategy showing 15% a year
gross can be negative once statutory charges, slippage and impact are applied, and the penalty is
not uniform — it scales with turnover, so it falls hardest on exactly the high-churn strategies that
look best on paper.

## What is statutory and what is an assumption

Two very different kinds of number live in this module and they must never be confused.

**Statutory and exchange charges are looked up, not modelled.** STT, NSE transaction charges, the
NSE Investor Protection Fund Trust levy, SEBI turnover fees, stamp duty and GST are published rates.
Every one is sourced in `config.yaml` with its effective date and circular reference. They are
time-varying — Union Budgets and SEBI circulars move them — so `config.yaml` holds a **schedule**
of dated epochs and every charge is priced at the rates in force **on its own trade date**. Pricing
a 2020 fill at 2026 rates is an anachronism that produces perfectly healthy-looking numbers, which
is why it is structurally impossible here: :meth:`CostModel.charge_leg` takes the date.

**Slippage and impact are assumptions.** They are not published, not measured here, and not
calibrated to Indian data — they are engineering defaults, and must never be described otherwise.
Their functional forms are standard; their coefficients are guesses that happen to be reasonable.
Every result that depends on them inherits that uncertainty, and the docstrings say so at each
point of use.

## Charge structure, equity delivery, per leg

    STT                 0.1% both sides
    Exchange charge     0.00297% both sides          (NSE cash, from 2024-10-01)
    IPFT                0.0001% both sides           (from 2023-04-01)
    SEBI turnover fee   0.0001% both sides
    Stamp duty          0.015% BUY ONLY              (uniform regime, from 2020-07-01)
    GST                 18% of (brokerage + exchange + IPFT + SEBI)
    DP charge           flat per scrip, SELL ONLY, not proportional to value

Note that GST applies to the *charges*, not to STT or stamp duty, which are themselves taxes.

## Why the sell leg is cheaper than the buy leg

Stamp duty is levied on purchase only, so a round trip is asymmetric. Code that charges a single
"round-trip cost" and halves it gets both legs wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from src.common.config import CostsConfig, RateSchedule

Side = Literal["buy", "sell"]
Segment = Literal["delivery", "intraday"]


@dataclass(frozen=True)
class LegCost:
    """Every component of the cost of one trade leg, in rupees.

    Held broken out rather than summed because a total alone cannot be checked against a broker's
    calculator, and checking it against one is the Phase 1.1 verification the PI performs by hand.
    """

    turnover: float
    brokerage: float
    stt: float
    exchange_charge: float
    ipft: float
    sebi_fee: float
    stamp_duty: float
    gst: float
    dp_charge: float

    @property
    def statutory_total(self) -> float:
        """Everything that is a published rate. Excludes slippage and impact, which are modelled."""
        return (
            self.brokerage + self.stt + self.exchange_charge + self.ipft
            + self.sebi_fee + self.stamp_duty + self.gst + self.dp_charge
        )

    @property
    def fraction_of_turnover(self) -> float:
        return self.statutory_total / self.turnover if self.turnover else 0.0

    def breakdown(self) -> str:
        """One line per component, for the checkpoint report and for eyeballing against a broker."""
        rows = [
            ("turnover", self.turnover), ("brokerage", self.brokerage), ("STT", self.stt),
            ("exchange charge", self.exchange_charge), ("IPFT", self.ipft),
            ("SEBI turnover fee", self.sebi_fee), ("stamp duty", self.stamp_duty),
            ("GST", self.gst), ("DP charge", self.dp_charge),
        ]
        width = max(len(name) for name, _ in rows)
        lines = [f"  {name:<{width}}  {value:>12,.4f}" for name, value in rows]
        lines.append(f"  {'TOTAL':<{width}}  {self.statutory_total:>12,.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ExecutionCost:
    """Statutory charges plus the modelled execution frictions for one leg."""

    leg: LegCost
    slippage: float
    impact: float

    @property
    def total(self) -> float:
        return self.leg.statutory_total + self.slippage + self.impact

    @property
    def fraction_of_turnover(self) -> float:
        return self.total / self.leg.turnover if self.leg.turnover else 0.0


class CostModel:
    """Charges one trade leg at a time, from rates held in ``config.yaml``.

    Stateless and cheap; a backtest constructs one and calls it per leg per session.
    """

    def __init__(self, config: CostsConfig) -> None:
        self._cfg = config

    def schedule_summary(self) -> str:
        """One line per rate epoch, for run manifests and checkpoint reports."""
        return "\n".join(
            f"  from {entry.effective_from}  exchange {entry.exchange_transaction_charge:.7%}  "
            f"STT delivery {entry.delivery.stt_buy:.3%}/{entry.delivery.stt_sell:.3%}  "
            f"({entry.label})"
            for entry in self._cfg.schedule
        )

    def _brokerage(self, turnover: float, segment: Segment, rates: RateSchedule) -> float:
        """Discount-broker brokerage: zero on delivery, the lesser of a rate and a cap intraday."""
        broker = rates.brokerage
        if segment == "delivery":
            return min(turnover * broker.delivery_rate, broker.delivery_flat) \
                if broker.delivery_flat else turnover * broker.delivery_rate
        return min(turnover * broker.intraday_rate, broker.intraday_flat_cap)

    def charge_leg(
        self,
        turnover: float,
        side: Side,
        segment: Segment = "delivery",
        n_scrips: int = 0,
        on: date | None = None,
    ) -> LegCost:
        """Statutory and exchange charges for a single leg of ``turnover`` rupees.

        ``on`` is the **trade date**, and it selects the rate epoch. Omitting it prices at the most
        recent epoch, which is right for a broker-calculator comparison and wrong for a backtest —
        so the engine always passes the fill session, and never relies on the default.

        ``n_scrips`` is the number of distinct symbols sold, used only for the flat depository
        charge, which is per scrip rather than per rupee and is levied on the sell side alone.
        """
        if turnover < 0:
            raise ValueError(f"turnover must be non-negative, got {turnover}")
        cfg = self._cfg
        epoch = cfg.schedule[-1] if on is None else cfg.rates_on(on)
        rates = epoch.delivery if segment == "delivery" else epoch.intraday

        brokerage = self._brokerage(turnover, segment, epoch)
        stt = turnover * (rates.stt_buy if side == "buy" else rates.stt_sell)
        stamp = turnover * (rates.stamp_duty_buy if side == "buy" else rates.stamp_duty_sell)
        exchange = turnover * epoch.exchange_transaction_charge
        ipft = turnover * epoch.ipft_charge
        sebi = turnover * epoch.sebi_turnover_fee

        # GST applies to the service charges only. STT and stamp duty are taxes in their own right
        # and are not themselves taxed.
        gst = (brokerage + exchange + ipft + sebi) * epoch.gst_rate

        dp = cfg.dp_charge_per_scrip_sell * n_scrips if side == "sell" else 0.0

        return LegCost(
            turnover=turnover, brokerage=brokerage, stt=stt, exchange_charge=exchange,
            ipft=ipft, sebi_fee=sebi, stamp_duty=stamp, gst=gst, dp_charge=dp,
        )

    def slippage(self, turnover: float, day_traded_value: float) -> float:
        """Volume-participation slippage: cost rises with the share of the day's volume consumed.

        ``slippage = turnover * coefficient * participation_rate``

        **A flat basis-point assumption is explicitly not acceptable** — it makes a large order look
        as cheap as a small one, which is the single most common way a backtest overstates capacity.

        **The coefficient is an assumption**, not an Indian measurement. The *shape* — linear in
        participation — is the defensible part; the level is not calibrated.
        """
        if day_traded_value <= 0:
            # No traded value means no basis for an estimate. Charging the cap is the conservative
            # reading and is preferable to charging zero, which would silently reward illiquidity.
            return turnover * self._cfg.slippage.max_slippage_fraction
        participation = turnover / day_traded_value
        fraction = min(
            self._cfg.slippage.participation_coefficient * participation,
            self._cfg.slippage.max_slippage_fraction,
        )
        return turnover * fraction

    def impact(
        self, turnover: float, day_traded_value: float, volatility: float | None = None
    ) -> float:
        """Square-root market impact: ``impact = turnover * c * sigma * sqrt(participation)``.

        **ASSUMPTION.** The square-root form is standard in the microstructure literature (Almgren,
        Thum, Hauptmann and Li, 2005) and is used here for its shape. The coefficient is *not*
        calibrated to Indian equities, and the daily-bar data available here could not calibrate it
        — that needs intraday volume profiles. Any capacity conclusion drawn from this is
        provisional and must say so.
        """
        if day_traded_value <= 0 or turnover <= 0:
            return 0.0
        sigma = self._cfg.impact.default_volatility if volatility is None else volatility
        participation = turnover / day_traded_value
        return turnover * self._cfg.impact.coefficient * sigma * math.sqrt(participation)

    def execute(
        self,
        turnover: float,
        side: Side,
        day_traded_value: float,
        segment: Segment = "delivery",
        n_scrips: int = 0,
        volatility: float | None = None,
        on: date | None = None,
    ) -> ExecutionCost:
        """Full cost of one leg: statutory charges, plus slippage, plus impact."""
        return ExecutionCost(
            leg=self.charge_leg(turnover, side, segment, n_scrips, on),
            slippage=self.slippage(turnover, day_traded_value),
            impact=self.impact(turnover, day_traded_value, volatility),
        )

    def round_trip(self, turnover: float, segment: Segment = "delivery",
                   n_scrips: int = 0, on: date | None = None) -> tuple[LegCost, LegCost]:
        """Buy and sell legs of the same notional. Returned separately: they are not symmetric.

        Stamp duty is charged on the buy alone and the depository charge on the sell alone, so a
        halved round-trip figure is wrong for both legs.
        """
        return (
            self.charge_leg(turnover, "buy", segment, 0, on),
            self.charge_leg(turnover, "sell", segment, n_scrips, on),
        )
