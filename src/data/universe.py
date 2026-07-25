"""Point-in-time investable universe, reconstructed from traded value alone.

**Why this is not an index membership list.** No free source publishes historical NIFTY
constituents with dated changes, and applying today's constituent list backwards is
survivorship-biased — it would quietly delete every company that later failed, which inflates
every downstream result and would invalidate the whole lab.

So the universe is *derived* instead of *looked up*. At each rebalance date the constituents are
the highest-ranked equities by median daily traded value over the preceding window, computed from
sessions strictly before that date. This is survivorship-free by construction: nothing consults a
list that knows the future, and a company that later delists simply stops appearing once its
trading stops, exactly as a contemporaneous investor would have experienced.

The cost of this choice is that the result is a liquidity proxy, **not** the NIFTY 100. Nothing
downstream may describe it as such.

Buffer rule: a name must reach ``entry_rank`` to join, and an incumbent is dropped only once it
falls past ``exit_rank``. Without that hysteresis, stocks sitting near the cutoff would churn in
and out at every rebalance, manufacturing turnover that costs real money in the Phase 1.1 model
while carrying no information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.common.config import UniverseConfig
from src.common.exceptions import DataIntegrityError, PointInTimeError
from src.common.log import get_logger
from src.data.calendar import TradingCalendar

_log = get_logger(__name__)


@dataclass(frozen=True)
class UniverseSnapshot:
    """The investable set effective from ``rebalance_date`` until the next rebalance."""

    rebalance_date: date
    symbols: tuple[str, ...]
    # Kept for auditability: how each name ranked when it was selected.
    ranks: dict[str, int]

    def __len__(self) -> int:
        return len(self.symbols)


def rebalance_dates(
    calendar: TradingCalendar,
    start: date,
    end: date,
    frequency: str,
) -> list[date]:
    """First trading session of each period in [start, end].

    Quarterly mirrors real index review practice and keeps turnover — and therefore cost — well
    below a monthly schedule.
    """
    if frequency not in ("quarterly", "monthly"):
        raise ValueError(f"unsupported rebalance frequency: {frequency}")
    step = 3 if frequency == "quarterly" else 1

    sessions = calendar.sessions_in_range(start, end)
    if not sessions:
        raise DataIntegrityError(f"no trading sessions between {start} and {end}")

    seen: set[tuple[int, int]] = set()
    picked: list[date] = []
    for session in sessions:
        # Bucket each session into its period; the first session seen for a bucket is the rebalance.
        period = (session.year, (session.month - 1) // step)
        if period not in seen:
            seen.add(period)
            picked.append(session)
    return picked


def _liquidity_table(panel: pl.DataFrame, window: list[date]) -> pl.DataFrame:
    """Median traded value and session count per symbol over the given sessions."""
    return (
        panel.filter(pl.col("session_date").is_in(window))
        .group_by("symbol")
        .agg(
            pl.col("turnover_inr").median().alias("median_turnover"),
            pl.len().alias("sessions_traded"),
        )
    )


def select_universe(
    panel: pl.DataFrame,
    calendar: TradingCalendar,
    as_of: date,
    cfg: UniverseConfig,
    incumbents: frozenset[str] = frozenset(),
) -> UniverseSnapshot:
    """Choose the universe effective on ``as_of`` using only sessions strictly before it.

    Args:
        panel: stacked bhavcopy rows with session_date, symbol, turnover_inr.
        calendar: trading calendar, used to define the trailing window in sessions.
        as_of: the rebalance date; no data stamped on or after this date may be consulted.
        cfg: universe parameters (window length, size, buffer bands, eligibility floors).
        incumbents: symbols currently held, which the buffer rule protects until they fall
            past ``exit_rank``.

    Raises:
        PointInTimeError: if the supplied panel contains rows at or after ``as_of``. The lookback
            is enforced here rather than trusted to callers, because a universe built with even
            one day of future data silently contaminates everything downstream.
    """
    future = panel.filter(pl.col("session_date") >= as_of)
    if future.height:
        raise PointInTimeError(
            f"universe selection for {as_of} was given {future.height} rows dated on or after it; "
            "the trailing window must end strictly before the rebalance date"
        )

    prior = calendar.sessions_in_range(calendar.first_session, as_of)
    window = [s for s in prior if s < as_of][-cfg.trailing_sessions:]
    if len(window) < cfg.trailing_sessions:
        raise DataIntegrityError(
            f"only {len(window)} sessions available before {as_of}, "
            f"need {cfg.trailing_sessions}; extend the calendar history"
        )

    stats = _liquidity_table(panel, window)

    # Eligibility: enough listed history, and actually traded on most of it. A name that barely
    # trades can post a flattering median on a handful of sessions.
    min_sessions = max(cfg.min_listed_sessions * cfg.min_traded_fraction,
                       len(window) * cfg.min_traded_fraction)
    eligible = stats.filter(pl.col("sessions_traded") >= min_sessions)

    ranked = eligible.sort("median_turnover", descending=True).with_row_index("rank", offset=1)
    order = ranked["symbol"].to_list()
    rank_of = dict(zip(order, ranked["rank"].to_list(), strict=True))

    # Buffer, applied with explicit incumbent preference. Retaining an incumbent inside the wider
    # exit band is only meaningful if it also survives the trim to `size` — otherwise a newcomer
    # would evict it anyway and the hysteresis would do nothing. So protected incumbents claim
    # their slots first, and newcomers fill what remains.
    protected = [s for s in order if s in incumbents and rank_of[s] <= cfg.exit_rank]
    newcomers = [s for s in order if s not in incumbents and rank_of[s] <= cfg.entry_rank]

    chosen = protected[: cfg.size]
    for symbol in newcomers:
        if len(chosen) >= cfg.size:
            break
        chosen.append(symbol)

    # Backfill if the bands leave the universe short (early history, or a thin eligible set):
    # take the best remaining names by rank so the universe size stays constant over time.
    if len(chosen) < cfg.size:
        taken = set(chosen)
        for symbol in order:
            if len(chosen) >= cfg.size:
                break
            if symbol not in taken:
                chosen.append(symbol)

    if len(chosen) < cfg.size:
        raise DataIntegrityError(
            f"only {len(chosen)} eligible names on {as_of}, need {cfg.size}; "
            "the eligibility filter or the panel coverage is too strict"
        )
    chosen.sort(key=lambda s: rank_of[s])
    return UniverseSnapshot(as_of, tuple(chosen), {s: rank_of[s] for s in chosen})


def build_universe_history(
    panel: pl.DataFrame,
    calendar: TradingCalendar,
    start: date,
    end: date,
    cfg: UniverseConfig,
) -> list[UniverseSnapshot]:
    """Roll the selection forward across every rebalance, carrying incumbents for the buffer."""
    snapshots: list[UniverseSnapshot] = []
    incumbents: frozenset[str] = frozenset()
    for as_of in rebalance_dates(calendar, start, end, cfg.rebalance):
        visible = panel.filter(pl.col("session_date") < as_of)
        snapshot = select_universe(visible, calendar, as_of, cfg, incumbents)
        snapshots.append(snapshot)
        incumbents = frozenset(snapshot.symbols)
        _log.info("universe %s: %d names", as_of, len(snapshot))
    return snapshots


def snapshots_to_frame(snapshots: list[UniverseSnapshot]) -> pl.DataFrame:
    """Flatten snapshots into a tidy frame for storage and inspection."""
    return pl.DataFrame(
        {
            "rebalance_date": [s.rebalance_date for s in snapshots for _ in s.symbols],
            "symbol": [sym for s in snapshots for sym in s.symbols],
            "rank": [s.ranks[sym] for s in snapshots for sym in s.symbols],
        }
    )
