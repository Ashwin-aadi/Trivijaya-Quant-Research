"""Constraint-based deployment capacity: how much the strategy can trade, not what it earns.

**This is not a market-impact capacity model, and no figure it produces may be described as one.**
The distinction is the PI's ruling of 2026-08-03 and it is the difference between two questions:

*Impact-erosion capacity* asks at what AUM the strategy's own trading moves prices enough to erase
its edge. Answering it requires an impact function. Phase 3.0 measured whether daily bars can supply
one and found that they cannot, so this repository does not answer that question.

*Constraint-based deployment capacity* asks at what AUM the strategy can no longer place its trades
without exceeding a stated share of the day's traded value. It is arithmetic on observed turnover.
It assumes nothing, it is what a real desk's pre-trade risk system actually enforces, and it is a
strictly smaller claim than the first — which is why it can be made honestly.

The definition. A book with target weights ``w`` rebalancing to ``w'`` trades ``|w' - w|`` of the
portfolio in each name. At deployed AUM ``A`` the rupee trade in name ``i`` is ``A * |dw_i|``. The
participation constraint permits at most ``kappa`` of that session's traded value, so

    A_max(t) = min over traded names of  kappa * traded_value_i(t) / |dw_i(t)|

which is the largest AUM at which *every* leg of that session's rebalance remains executable. The
binding name is recorded, because capacity set by one illiquid position is a different fact from
capacity set by the book as a whole, and only the former is fixable by the portfolio manager.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.common.exceptions import DataIntegrityError


@dataclass(frozen=True)
class CapacitySummary:
    """A factor's deployable AUM in rupees, summarised across the sessions it rebalanced on.

    **The median was the headline until Checkpoint 3.3 and should not have been.** A strategy that
    holds a constant book has near-zero rebalancing turnover on almost every session, so its median
    session capacity is enormous — while the one session that builds the position is tightly
    constrained and is the session that actually determines how much money the strategy can run.
    Taking the median discarded exactly the binding constraint. The machine-generated corpus is full
    of such strategies; the five standard factor strategies, which rebalance substantially every
    session, contained none, so the defect was invisible until the benchmark met its intended
    population.

    ``binding_capacity_inr`` is therefore the headline and ``median_capacity_inr`` is retained as a
    typical-session statistic. The 5th percentile sits between them: a strategy deployable at a
    billion rupees on a typical day and ten million on a bad one has a capacity of ten million for
    any desk that cannot skip its bad days.
    """

    factor: str
    #: **The headline figure.** The largest AUM at which *every* session the strategy trades stays
    #: inside the participation limit, including the session that builds the opening position. A
    #: strategy must be executable on all of its rebalances, not on a typical one, so this is the
    #: capacity a desk could actually run. It also makes initial construction binding for a
    #: near-static strategy, which is what the median could not do — see the class note below.
    binding_capacity_inr: float
    #: Capacity on the strategy's first session: the cost of acquiring the book at all.
    entry_capacity_inr: float
    median_capacity_inr: float
    p05_capacity_inr: float
    p95_capacity_inr: float
    n_rebalance_sessions: int
    participation_limit: float
    #: Share of sessions whose capacity was set by a single name rather than by the book. High
    #: values mean the number is a statement about one position, not about the strategy.
    fraction_bound_by_one_name: float
    #: **Robustness check, not an alternative definition** (PI ruling, 2026-08-03). What capacity
    #: would be if the single tightest position were excluded each session. Reported beside the
    #: constrained figure to show whether that figure describes the strategy or one holding; it must
    #: never be reported alone, and never as the headline. A desk cannot in general simply drop its
    #: least liquid name — doing so changes the portfolio, and therefore changes the returns the
    #: capacity is a capacity *for*.
    median_relaxed_capacity_inr: float
    relaxed_over_constrained: float


def turnover_by_session(
    weights: pl.DataFrame, *, min_traded_fraction: float = 0.0
) -> pl.DataFrame:
    """Per name and session, the fraction of the book that must trade to reach the target weights.

    A name entering the book trades its full weight — **the strategy's first session therefore
    counts as the full acquisition of its opening position**, which is what makes initial portfolio
    construction part of deployment capacity rather than a cost the measure ignores. A name leaving
    the book trades its full previous weight, which the outer join is there to catch; dropping
    departures would understate turnover and so overstate capacity, invisibly.

    ``min_traded_fraction`` discards weight changes too small to be trades. Without it, a strategy
    that recomputes an unchanged weight each session produces differences of order 1e-18, and
    dividing a participation limit by those reports a capacity of 1e23 crore. Added by PI ruling at
    Checkpoint 3.3, after the machine-generated corpus exercised a path the five standard factor
    strategies never reached.
    """
    required = {"session_date", "factor", "symbol", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise DataIntegrityError(f"weights frame is missing columns {sorted(missing)}")

    sessions = weights["session_date"].unique().sort()
    previous = weights.with_columns(
        session_date=pl.col("session_date").replace_strict(
            old=sessions.to_list()[:-1], new=sessions.to_list()[1:], default=None
        )
    ).drop_nulls("session_date")

    return (
        weights.join(
            previous.rename({"weight": "prev_weight"}),
            on=["session_date", "factor", "symbol"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            weight=pl.col("weight").fill_null(0.0),
            prev_weight=pl.col("prev_weight").fill_null(0.0),
        )
        .with_columns(traded_fraction=(pl.col("weight") - pl.col("prev_weight")).abs())
        .filter(pl.col("traded_fraction") > max(min_traded_fraction, 0.0))
        .select(["session_date", "factor", "symbol", "traded_fraction"])
    )


def session_capacity(
    traded: pl.DataFrame,
    liquidity: pl.DataFrame,
    *,
    participation_limit: float,
) -> pl.DataFrame:
    """Largest AUM at which every leg of each session's rebalance stays inside the limit.

    ``liquidity`` supplies ``adv_inr`` per symbol-session, which must be a *trailing* average — a
    capacity computed against the same session's own traded value would be sizing the order using
    the volume that the order itself helped create.
    """
    if participation_limit <= 0:
        raise DataIntegrityError(f"participation limit must be positive, got {participation_limit}")
    joined = (
        traded.join(liquidity, on=["session_date", "symbol"], how="inner")
        .drop_nulls("adv_inr")
        .filter(pl.col("adv_inr") > 0)
        .with_columns(
            name_capacity=participation_limit * pl.col("adv_inr") / pl.col("traded_fraction")
        )
    )
    return (
        joined.group_by(["factor", "session_date"])
        .agg(
            capacity_inr=pl.col("name_capacity").min(),
            binding_symbol=pl.col("symbol").sort_by("name_capacity").first(),
            n_names=pl.len(),
            # slice rather than gather: a session with a single traded name has no second-tightest
            # constraint, and gather would raise on it where slice correctly yields null.
            second_capacity=pl.col("name_capacity").sort().slice(1, 1).first(),
        )
        .sort(["factor", "session_date"])
    )


def summarise_capacity(
    per_session: pl.DataFrame,
    *,
    participation_limit: float,
    single_name_ratio: float = 2.0,
) -> list[CapacitySummary]:
    """Collapse per-session capacity into one summary per factor.

    A session counts as "bound by one name" when the tightest name is more than
    ``single_name_ratio`` times tighter than the next tightest — that is, when dropping one
    position would have raised capacity by more than that factor.
    """
    summaries: list[CapacitySummary] = []
    for (factor,), group in per_session.group_by(["factor"], maintain_order=True):
        capacities = group["capacity_inr"].to_numpy()
        concentrated = group.filter(
            pl.col("second_capacity").is_not_null()
            & (pl.col("second_capacity") > single_name_ratio * pl.col("capacity_inr"))
        )
        # The relaxed figure is the second-tightest name: capacity had the tightest been dropped.
        # Sessions with only one traded name have no second constraint and are excluded rather than
        # given an infinite one.
        relaxed = group["second_capacity"].drop_nulls().to_numpy()
        median = float(np.median(capacities))
        first_session = group.sort("session_date").head(1)
        median_relaxed = float(np.median(relaxed)) if relaxed.size else float("nan")
        summaries.append(
            CapacitySummary(
                factor=str(factor),
                binding_capacity_inr=float(capacities.min()),
                entry_capacity_inr=float(first_session["capacity_inr"][0]),
                median_capacity_inr=median,
                p05_capacity_inr=float(np.quantile(capacities, 0.05)),
                p95_capacity_inr=float(np.quantile(capacities, 0.95)),
                n_rebalance_sessions=group.height,
                participation_limit=participation_limit,
                fraction_bound_by_one_name=concentrated.height / group.height,
                median_relaxed_capacity_inr=median_relaxed,
                relaxed_over_constrained=median_relaxed / median if median else float("nan"),
            )
        )
    return summaries


def capacity_by_flow_state(
    per_session: pl.DataFrame,
    flow_states: pl.DataFrame,
) -> pl.DataFrame:
    """Median deployable AUM per factor per flow state, with the session count for each.

    The count matters as much as the median here: an outflow-state capacity computed over thirty
    sessions is not evidence of anything, and the ratio between states is the headline that a reader
    will quote, so it must never appear without the two sample sizes that produced it.
    """
    if "flow_state" not in flow_states.columns:
        raise DataIntegrityError("flow-state frame needs a flow_state column")
    return (
        per_session.join(
            flow_states.select(["session_date", "flow_state"]), on="session_date", how="inner"
        )
        .drop_nulls("flow_state")
        .group_by(["factor", "flow_state"])
        .agg(
            median_capacity_inr=pl.col("capacity_inr").median(),
            p05_capacity_inr=pl.col("capacity_inr").quantile(0.05),
            n_sessions=pl.len(),
        )
        .sort(["factor", "flow_state"])
    )
