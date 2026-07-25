"""Ingest NSE daily bhavcopy (all-stock OHLCV) into immutable per-session raw files.

This is the backbone of the whole data layer: the universe is ranked from the traded values in
here, and prices are validated against it.

**The silent-corruption hazard this module exists to defeat.** For dates NSE no longer serves,
the upstream helper writes the *HTML error page* to a file named `.csv` and returns normally. A
loop that trusts the return value produces thousands of "successful" files full of markup, and
naive row-count checks pass. Every file is therefore content-validated here before it is accepted:
non-CSV content raises, it is never skipped, logged-and-ignored, or partially salvaged.

One parquet per session is written under ``data/raw/bhavcopy/``. That granularity is deliberate —
it makes the download resumable, so an interruption costs one session rather than the whole run,
and it keeps each file immutable in the sense RULE 6 requires.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet, write_raw_parquet
from src.common.log import get_logger

_log = get_logger(__name__)

# Columns we require to be present after whitespace stripping. NSE ships the header with leading
# spaces (" SERIES"), which silently breaks naive key lookups, so names are normalised on read.
REQUIRED_COLUMNS = frozenset(
    {"SYMBOL", "SERIES", "CLOSE_PRICE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
     "PREV_CLOSE", "TTL_TRD_QNTY", "TURNOVER_LACS"}
)

# Only ordinary rolling-settlement equity counts as investable here. BE/SM/ST and the rest are
# trade-to-trade, SME, or other segments with different microstructure.
EQUITY_SERIES = "EQ"


def session_path(raw_root: Path, session: date) -> Path:
    """Where one session's validated bhavcopy lives."""
    return raw_root / "bhavcopy" / f"cm_{session:%Y%m%d}.parquet"


def _looks_like_html(first_line: str) -> bool:
    """True if the payload is a web page rather than CSV — the silent failure described above."""
    head = first_line.lstrip().lower()
    return head.startswith("<!doctype") or head.startswith("<html") or "<html" in head


def parse_bhavcopy_csv(csv_path: Path, session: date) -> pl.DataFrame:
    """Read and validate one bhavcopy CSV, returning only its equity rows.

    Raises DataIntegrityError rather than returning a partial or empty frame — a bad day must stop
    the pipeline loudly instead of quietly shrinking the universe on that date.
    """
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as handle:
        first_line = handle.readline()
        if not first_line.strip():
            raise DataIntegrityError(f"bhavcopy for {session} is empty: {csv_path}")
        if _looks_like_html(first_line):
            raise DataIntegrityError(
                f"bhavcopy for {session} is an HTML page, not CSV — NSE does not serve this date "
                f"({csv_path})"
            )
        handle.seek(0)
        # Normalise header whitespace up front; NSE's leading spaces are a known footgun.
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataIntegrityError(f"bhavcopy for {session} has no header row: {csv_path}")
        fields = [name.strip() for name in reader.fieldnames]
        missing = REQUIRED_COLUMNS - set(fields)
        if missing:
            raise DataIntegrityError(
                f"bhavcopy for {session} is missing columns {sorted(missing)}; got {fields}"
            )
        rows = [{k.strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in raw_row.items() if k is not None}
                for raw_row in reader]

    equity = [r for r in rows if r.get("SERIES") == EQUITY_SERIES]
    if not equity:
        raise DataIntegrityError(
            f"bhavcopy for {session} parsed but contains zero {EQUITY_SERIES} rows "
            f"({len(rows)} total rows) — treating as corrupt rather than an empty market"
        )

    frame = pl.DataFrame(
        {
            "session_date": [session] * len(equity),
            "symbol": [r["SYMBOL"] for r in equity],
            "open": [float(r["OPEN_PRICE"]) for r in equity],
            "high": [float(r["HIGH_PRICE"]) for r in equity],
            "low": [float(r["LOW_PRICE"]) for r in equity],
            "close": [float(r["CLOSE_PRICE"]) for r in equity],
            "prev_close": [float(r["PREV_CLOSE"]) for r in equity],
            "volume": [float(r["TTL_TRD_QNTY"]) for r in equity],
            # NSE reports turnover in lakhs; convert to rupees so every money figure in the repo
            # is in the same unit and no downstream code has to remember this.
            "turnover_inr": [float(r["TURNOVER_LACS"]) * 100_000 for r in equity],
        }
    )
    if frame["close"].min() is not None and frame["close"].min() <= 0:  # type: ignore[operator]
        raise DataIntegrityError(f"bhavcopy for {session} contains a non-positive close price")
    return frame


def fetch_session(raw_root: Path, session: date) -> pl.DataFrame:
    """Download, validate, and immutably cache one session. Returns the cached frame if present."""
    target = session_path(raw_root, session)
    if target.exists():
        return read_parquet(target)

    from jugaad_data.nse import bhavcopy_save  # lazy: keeps parsing importable without network

    scratch = Path(tempfile.mkdtemp(prefix="bhav_"))
    try:
        downloaded = Path(bhavcopy_save(session, str(scratch)))
        frame = parse_bhavcopy_csv(downloaded, session)
        write_raw_parquet(
            frame,
            target,
            source_url="https://nsearchives.nseindia.com (NSE daily bhavcopy)",
            extra={"session_date": session.isoformat(), "series_filter": EQUITY_SERIES},
        )
        return frame
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
