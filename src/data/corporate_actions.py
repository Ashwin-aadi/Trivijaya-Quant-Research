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


def fetch_declared_splits(symbol: str, start: date, end: date) -> list[AdjustmentEvent]:
    """Corporate actions as declared by an independent source, for one symbol.

    Necessary because the restated-previous-close method above is unreliable: NSE leaves
    ``PREV_CLOSE`` un-restated for a large share of real capital changes, so detection alone
    misses them and leaves fictitious crashes in the return series. A source that publishes split
    ratios directly does not depend on the exchange having restated anything.
    """
    import yfinance as yf

    series = yf.Ticker(f"{symbol}.NS").splits
    if series is None or series.empty:
        return []
    events: list[AdjustmentEvent] = []
    for stamp, ratio in series.items():
        ex_date = stamp.date()
        if start <= ex_date <= end and float(ratio) > 0:
            events.append(AdjustmentEvent(symbol=symbol, ex_date=ex_date, factor=float(ratio)))
    return events


@dataclass(frozen=True)
class FactorDisagreement:
    """A declared split ratio that the observed price move does not corroborate."""

    symbol: str
    ex_date: date
    declared_factor: float
    implied_factor: float      # prior close divided by ex-date close


def reconcile_with_prices(
    events: list[AdjustmentEvent],
    panel: pl.DataFrame,
    calendar: TradingCalendar,
    rel_tol: float = 0.10,
) -> tuple[list[AdjustmentEvent], list[FactorDisagreement]]:
    """Check each declared factor against the price move actually observed on its ex-date.

    A genuine 1:2 split roughly halves the quoted price, so the declared ratio and the observed
    ratio should agree closely. Where they do not, the event is usually not a clean split — a
    demerger, or a split combined with a real move — and silently applying the declared factor
    would inject an error rather than remove one. Those cases are returned separately so they can
    be reported and judged, never quietly applied.
    """
    prices = {
        (row["symbol"], row["session_date"]): row["close"]
        for row in panel.select(["symbol", "session_date", "close"]).iter_rows(named=True)
    }
    agreed: list[AdjustmentEvent] = []
    disputed: list[FactorDisagreement] = []

    for event in events:
        if not calendar.is_trading_day(event.ex_date):
            continue
        try:
            previous = calendar.previous_session(event.ex_date)
        except Exception:  # noqa: BLE001 - outside calendar range; nothing to reconcile against
            continue
        before = prices.get((event.symbol, previous))
        on_day = prices.get((event.symbol, event.ex_date))
        if not before or not on_day or on_day <= 0:
            continue
        implied = before / on_day
        if abs(implied - event.factor) / event.factor <= rel_tol:
            agreed.append(event)
        else:
            disputed.append(
                FactorDisagreement(event.symbol, event.ex_date, event.factor, implied)
            )

    _log.info("reconciled %d declared events: %d corroborated by prices, %d disputed",
              len(events), len(agreed), len(disputed))
    return agreed, disputed


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
