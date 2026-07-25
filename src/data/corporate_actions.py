"""Detect split/bonus adjustments from the exchange's own restated previous-close figures.

**How this works, and why it is trustworthy.** On the ex-date of a split, bonus, or similar
capital change, NSE restates that session's ``PREV_CLOSE`` onto the post-event basis while the
prior session's ``CLOSE_PRICE`` remains on the old basis. The ratio between them is therefore the
exchange's own adjustment factor, taken from the price file rather than from a third-party
corporate-actions feed that may disagree with it. A 1:1 bonus shows up as a ratio near 2.0, a 1:5
split as a ratio near 5.0.

**The trap this module is built around.** The ratio is only meaningful between two *consecutive
trading sessions*. If the panel has a hole — a failed download, a partial backfill — then the
"previous" row silently comes from days or years earlier and the ratio becomes a large, entirely
fictitious number that looks exactly like a stock split. Detection therefore refuses to run on a
panel with gaps rather than emitting plausible nonsense.

**Known limitation, stated rather than buried.** NSE restates the previous close for capital
changes but generally not for ordinary cash dividends, so the factors recovered here cover splits,
bonuses, and similar events. Ordinary dividends are not captured, which means the adjusted series
is a price-return series, not a total-return series. Anything comparing against a total-return
benchmark must account for that difference explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from src.common.exceptions import DataIntegrityError
from src.common.log import get_logger
from src.data.calendar import TradingCalendar

_log = get_logger(__name__)

# Below this, a discrepancy is rounding or a tick convention rather than a capital change.
# A 5% move in the restated previous close is far larger than any rounding artifact but well
# under the smallest realistic split or bonus ratio.
MIN_ADJUSTMENT_RATIO = 1.05


@dataclass(frozen=True)
class AdjustmentEvent:
    """A restated previous close, i.e. an ex-date for a split, bonus, or similar."""

    symbol: str
    ex_date: date
    factor: float          # prior session's close divided by the restated previous close


def assert_panel_contiguous(panel: pl.DataFrame, calendar: TradingCalendar) -> None:
    """Refuse to proceed if the panel is missing sessions the calendar says the market was open.

    A hole here would make every ratio spanning it look like a corporate action, so this is a
    correctness precondition rather than a convenience check.
    """
    present = set(panel["session_date"].unique().to_list())
    if not present:
        raise DataIntegrityError("panel contains no sessions")
    expected = calendar.sessions_in_range(min(present), max(present))
    missing = sorted(set(expected) - present)
    if missing:
        raise DataIntegrityError(
            f"panel is missing {len(missing)} trading sessions between {min(present)} and "
            f"{max(present)} (first few: {missing[:5]}); corporate-action detection would "
            "report fictitious splits across the gaps"
        )


def detect_adjustments(
    panel: pl.DataFrame,
    calendar: TradingCalendar,
    min_ratio: float = MIN_ADJUSTMENT_RATIO,
) -> list[AdjustmentEvent]:
    """Find every ex-date in the panel by comparing consecutive sessions per symbol."""
    assert_panel_contiguous(panel, calendar)

    ordered = panel.sort(["symbol", "session_date"]).with_columns(
        pl.col("close").shift(1).over("symbol").alias("prior_close"),
        pl.col("session_date").shift(1).over("symbol").alias("prior_session"),
    )
    sessions = calendar.sessions_in_range(calendar.first_session, calendar.last_session)
    index_of = {s: i for i, s in enumerate(sessions)}

    candidates = (
        ordered.drop_nulls(["prior_close", "prior_session"])
        .filter(pl.col("prev_close") > 0)
        .with_columns((pl.col("prior_close") / pl.col("prev_close")).alias("ratio"))
        .filter((pl.col("ratio") >= min_ratio) | (pl.col("ratio") <= 1 / min_ratio))
    )

    events: list[AdjustmentEvent] = []
    for row in candidates.iter_rows(named=True):
        # Only adjacent sessions are comparable. A symbol that simply did not trade for a stretch
        # would otherwise produce a bogus factor spanning the quiet period.
        if index_of[row["session_date"]] - index_of[row["prior_session"]] != 1:
            continue
        events.append(
            AdjustmentEvent(
                symbol=row["symbol"],
                ex_date=row["session_date"],
                factor=float(row["ratio"]),
            )
        )
    _log.info("detected %d adjustment events across %d symbols",
              len(events), len({e.symbol for e in events}))
    return events


def adjustment_factors(
    events: list[AdjustmentEvent],
    symbol: str,
    sessions: list[date],
) -> dict[date, float]:
    """Cumulative divisor per session that puts an older raw price on today's basis.

    A price observed before an ex-date is quoted on the pre-event basis, so it must be divided by
    every factor that took effect after it. Walking backwards from the most recent session keeps
    the latest prices unchanged and restates history onto the current basis, which is the
    convention that makes a return series continuous across the event.
    """
    relevant = sorted((e for e in events if e.symbol == symbol), key=lambda e: e.ex_date)
    factors: dict[date, float] = {}
    cumulative = 1.0
    upcoming = {e.ex_date: e.factor for e in relevant}
    for session in sorted(sessions, reverse=True):
        factors[session] = cumulative
        # An event dated on session S restates everything strictly before S.
        if session in upcoming:
            cumulative *= upcoming[session]
    return factors


def apply_adjustments(
    panel: pl.DataFrame,
    events: list[AdjustmentEvent],
) -> pl.DataFrame:
    """Return the panel with split/bonus-adjusted OHLC columns added.

    Volume is scaled the opposite way, so that price times volume — the traded value the universe
    ranks on — is left unchanged by a purely cosmetic capital change.
    """
    if not events:
        return panel.with_columns(
            *[pl.col(c).alias(f"adj_{c}") for c in ("open", "high", "low", "close")],
            pl.col("volume").alias("adj_volume"),
        )

    sessions = sorted(panel["session_date"].unique().to_list())
    lookup: dict[tuple[str, date], float] = {}
    for symbol in {e.symbol for e in events}:
        for session, factor in adjustment_factors(events, symbol, sessions).items():
            lookup[(symbol, session)] = factor

    divisor = pl.Series(
        "divisor",
        [lookup.get((sym, dt), 1.0)
         for sym, dt in zip(panel["symbol"], panel["session_date"], strict=True)],
    )
    return panel.with_columns(divisor).with_columns(
        *[(pl.col(c) / pl.col("divisor")).alias(f"adj_{c}")
          for c in ("open", "high", "low", "close")],
        (pl.col("volume") * pl.col("divisor")).alias("adj_volume"),
    )
