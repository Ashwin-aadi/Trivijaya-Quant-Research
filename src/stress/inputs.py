"""Assemble the index close series that regime features are built from.

Kept apart from :mod:`src.stress.regimes` so that module stays pure logic over arrays and frames,
testable without touching the filesystem — the same split :mod:`src.data.calendar` uses between
``TradingCalendar`` and ``fetch_index_sessions``.

The series is stitched from two raw files that abut without overlapping:

* ``calendar_cnx100_burnin.parquet`` — 2015-01-01 to 2018-12-31, fetched for Phase 2.0 solely as
  HMM fitting burn-in (PI-approved, DECISIONS.md Phase 2.0 decision 3).
* ``calendar_cnx100.parquet`` — 2019-01-01 onward, the Phase 1.0 calendar source.

The holdout calendar is deliberately **not** loaded here. Nothing in Phase 2.0 may read it.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.common.config import Config
from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet
from src.common.log import get_logger

_log = get_logger(__name__)

BURNIN_FILE = "calendar_cnx100_burnin.parquet"
CALENDAR_FILE = "calendar_cnx100.parquet"


def load_index_closes(cfg: Config, *, include_burnin: bool = True) -> pl.DataFrame:
    """Return ``session_date``/``close`` for the development window, optionally with burn-in ahead.

    Raises rather than silently proceeding if the two files overlap. An overlap would mean some
    sessions appear twice in the feature series, which would distort every trailing window and
    inflate the apparent training size — a failure that would otherwise be invisible in the output.
    """
    calendar_path = cfg.paths.data_raw / CALENDAR_FILE
    frames = [read_parquet(calendar_path).select("session_date", "close")]

    if include_burnin:
        burnin_path = cfg.paths.data_raw / BURNIN_FILE
        if not burnin_path.exists():
            raise DataIntegrityError(
                f"{burnin_path} is missing; run scripts/fetch_regime_burnin.py first. "
                "Fitting without it would leave the first refits with ~242 crisis-free sessions."
            )
        burnin = read_parquet(burnin_path).select("session_date", "close")
        overlap = set(burnin["session_date"].to_list()) & set(frames[0]["session_date"].to_list())
        if overlap:
            raise DataIntegrityError(
                f"burn-in and calendar share {len(overlap)} sessions "
                f"(e.g. {sorted(overlap)[:3]}); refusing to double-count them"
            )
        frames.insert(0, burnin)

    series = pl.concat(frames).sort("session_date").unique(subset="session_date", keep="first")
    series = series.sort("session_date")
    _log.info(
        "index series: %d sessions, %s .. %s",
        series.height,
        series["session_date"].min(),
        series["session_date"].max(),
    )
    return series


def burnin_only(series: pl.DataFrame, dev_start: date) -> pl.DataFrame:
    """Sessions strictly before the development window.

    This is the only data K may be selected on (PI, Phase 2.0 decision 5). Selecting K on any
    session a strategy is later scored over would be choosing a model parameter with sight of the
    evaluation period.
    """
    out = series.filter(pl.col("session_date") < dev_start)
    if out.height == 0:
        raise DataIntegrityError(f"no sessions before {dev_start}; nothing to select K on")
    return out
