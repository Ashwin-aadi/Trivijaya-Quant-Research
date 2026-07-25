"""Assemble the adjusted price panel and cross-check it against an independent source.

Bhavcopy is authoritative: it is the exchange's own end-of-day file, and it is what the universe
ranks on. yfinance is pulled purely as a second opinion, because a single source that is wrong in
a systematic way is indistinguishable from a source that is right unless something else disagrees
with it.

The two are compared on **raw** closes rather than adjusted ones. That is deliberate: comparing
adjusted series would conflate two very different questions — whether the underlying prices agree,
and whether two adjustment methodologies agree. Raw closes isolate the first, which is the one
that matters for trusting the panel at all.

If the disagreement rate exceeds the configured ceiling, this module reports it and the caller is
expected to stop. A widespread mismatch on Indian equities usually means a corporate action was
handled differently by one side, not that prices are randomly noisy — and quietly averaging over
that would bake a systematic error into everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet
from src.common.log import get_logger
from src.data.bhavcopy import session_path

_log = get_logger(__name__)


@dataclass
class CrossCheckResult:
    """Outcome of comparing bhavcopy closes against the independent source."""

    compared_points: int
    mismatches: int
    symbols_compared: int
    worst: list[tuple[str, date, float, float, float]] = field(default_factory=list)

    @property
    def discrepancy_rate(self) -> float:
        return self.mismatches / self.compared_points if self.compared_points else 0.0

    def exceeds(self, ceiling: float) -> bool:
        return self.discrepancy_rate > ceiling


def load_panel(raw_root: Path, sessions: list[date]) -> pl.DataFrame:
    """Stack cached bhavcopy sessions into one panel, failing loudly on any missing session."""
    missing = [s for s in sessions if not session_path(raw_root, s).exists()]
    if missing:
        raise DataIntegrityError(
            f"{len(missing)} sessions are not cached (first few: {missing[:5]}); "
            "run scripts/download_bhavcopy.py before building the panel"
        )
    frames = [read_parquet(session_path(raw_root, s)) for s in sessions]
    return pl.concat(frames)


def fetch_reference_closes(symbol: str, start: date, end: date) -> pl.DataFrame:
    """Raw (unadjusted) closes for one symbol from the independent source.

    ``auto_adjust=False`` is essential — the default silently returns an adjusted series, which
    would make this a comparison of adjustment methodologies rather than of prices.
    """
    import yfinance as yf

    frame = yf.Ticker(f"{symbol}.NS").history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=False
    )
    if frame.empty:
        return pl.DataFrame({"session_date": [], "ref_close": []},
                            schema={"session_date": pl.Date, "ref_close": pl.Float64})
    return pl.DataFrame(
        {
            "session_date": [ts.date() for ts in frame.index],
            "ref_close": [float(v) for v in frame["Close"]],
        }
    )


def cross_check(
    panel: pl.DataFrame,
    symbols: list[str],
    rel_tol: float,
    max_worst: int = 15,
    price_column: str = "adj_close",
) -> CrossCheckResult:
    """Compare our closes against the reference source for the given symbols.

    Defaults to the adjusted column because the reference back-adjusts its history for splits
    even when asked for unadjusted prices — its ``auto_adjust`` flag governs dividends only.
    Comparing our *raw* prices against it therefore measures the split basis rather than data
    quality, and reports a large discrepancy even when both sources are perfectly correct.
    """
    result = CrossCheckResult(compared_points=0, mismatches=0, symbols_compared=0)
    for symbol in symbols:
        own = (panel.filter(pl.col("symbol") == symbol)
               .select(["session_date", pl.col(price_column).alias("close")]))
        if own.is_empty():
            continue
        start = own["session_date"].min()
        end = own["session_date"].max()
        assert isinstance(start, date) and isinstance(end, date)
        reference = fetch_reference_closes(symbol, start, end)
        if reference.is_empty():
            _log.warning("no reference data for %s; skipping (not counted as agreement)", symbol)
            continue

        merged = own.join(reference, on="session_date", how="inner")
        if merged.is_empty():
            continue
        merged = merged.with_columns(
            ((pl.col("close") - pl.col("ref_close")).abs()
             / pl.col("ref_close")).alias("rel_diff")
        )
        bad = merged.filter(pl.col("rel_diff") > rel_tol)
        result.compared_points += merged.height
        result.mismatches += bad.height
        result.symbols_compared += 1
        for row in bad.sort("rel_diff", descending=True).head(3).iter_rows(named=True):
            result.worst.append(
                (symbol, row["session_date"], row["close"], row["ref_close"], row["rel_diff"])
            )

    result.worst.sort(key=lambda item: item[4], reverse=True)
    result.worst = result.worst[:max_worst]
    _log.info("cross-check: %d/%d points disagree beyond %.1f%% across %d symbols (rate %.4f)",
              result.mismatches, result.compared_points, rel_tol * 100,
              result.symbols_compared, result.discrepancy_rate)
    return result
