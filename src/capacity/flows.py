"""Participant-flow ingestion, and an honest account of what free sources actually serve.

**The headline data problem for Phase 3.0, stated before anything else.** The charter specifies
"FII/DII daily net flow ingestion from NSE public data". NSE publishes that series, at
``/api/fiidiiTradeReact``, and the endpoint **serves only the most recent session**. It accepts
``date``, ``from`` and ``to`` parameters and ignores all of them, returning the latest day
regardless — verified against three different parameter spellings, each of which returned the
identical current-day payload. BSE's equivalent API returned a generic error page for every
endpoint spelling attempted, and NSDL's FPI report rejected the request outright. Nothing in this
module can therefore build a historical daily FII/DII **cash-market** net-flow series, and nothing
in it pretends to.

What *is* freely available daily and historically, confirmed across 2020-01-02 to 2024-12-31:

``fao_participant_vol`` / ``fao_participant_oi``
    Participant-wise **equity-derivatives** activity, split into Client, DII, FII and Pro. This is
    the substitute this module implements, and it is a substitute rather than an equivalent. Three
    differences matter and none of them is small. It is denominated in **contracts, not rupees**,
    and contract values move with both price and lot size — SEBI revised index-derivative lot sizes
    materially in November 2024, inside the study window, so a contract count is not comparable
    across that boundary without a lot-size history this repository does not have. It describes
    **derivatives, not cash**, and the charter's own framing of India as a market where "F&O
    turnover far exceeds cash-market turnover" cuts both ways: the derivatives series is large and
    informative, and it is not the series whose effect on cash-market liquidity P3 set out to
    measure. And FII derivatives activity is substantially **hedging and arbitrage** against cash
    positions, so its sign need not track the direction of cash-market demand at all.

Whether that substitution is acceptable is a methodological fork, and it is put to the PI at
Checkpoint 3.0 rather than decided here.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
import requests

from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet, write_raw_parquet
from src.common.log import get_logger

_log = get_logger(__name__)

PARTICIPANT_VOLUME_URL = (
    "https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_{stamp}.csv"
)
PARTICIPANT_OI_URL = (
    "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{stamp}.csv"
)
#: Serves the current session only. Retained because it is the authoritative cash-market series and
#: a live cross-check against it is worth having, not because it can build history.
FII_DII_CASH_URL = "https://www.nseindia.com/api/fiidiiTradeReact"

#: The four participant categories NSE reports, plus the TOTAL row it appends. TOTAL is dropped on
#: ingestion: it is an identity over the others, and keeping it invites a downstream sum that
#: double-counts every contract.
PARTICIPANT_CATEGORIES = ("Client", "DII", "FII", "Pro")
_TOTAL_ROW = "TOTAL"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.nseindia.com/",
}


def _new_session() -> requests.Session:
    """A warmed HTTP session. NSE rejects unwarmed clients, which surfaces as an HTML error page."""
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    session.get("https://www.nseindia.com/reports/fii-dii", timeout=30)
    return session


def participant_path(raw_root: Path, session: date, kind: str) -> Path:
    """Where one session's participant file lives. ``kind`` is ``"vol"`` or ``"oi"``."""
    return raw_root / "participants" / f"fao_participant_{kind}_{session:%Y%m%d}.parquet"


def parse_participant_csv(text: str, session: date) -> pl.DataFrame:
    """Parse one participant-wise CSV into a tidy long frame, validating it is not an error page.

    NSE ships this file with the title row doubled-quoted, a stray tab inside two column names, and
    a layout that changed punctuation partway through the study window. Header names are therefore
    stripped of whitespace before use, and the file is validated by the presence of the four
    participant categories rather than by an exact header match — an exact match would break on a
    cosmetic change and pass on a substantive one.
    """
    if "<html" in text[:200].lower() or "<!doctype" in text[:200].lower():
        raise DataIntegrityError(
            f"participant file for {session} is an HTML page, not CSV; NSE does not serve this date"
        )
    lines = text.splitlines()
    if len(lines) < 3:
        raise DataIntegrityError(f"participant file for {session} has {len(lines)} lines, expected"
                                 " a title, a header and one row per category")
    reader = csv.DictReader(io.StringIO("\n".join(lines[1:])))
    if reader.fieldnames is None:
        raise DataIntegrityError(f"participant file for {session} has no header row")
    fields = [name.strip() for name in reader.fieldnames]
    rows = [
        {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
        for raw in reader
    ]
    by_category = {row[fields[0]]: row for row in rows if row.get(fields[0])}
    missing = set(PARTICIPANT_CATEGORIES) - set(by_category)
    if missing:
        raise DataIntegrityError(
            f"participant file for {session} is missing categories {sorted(missing)}; "
            f"got {sorted(by_category)}"
        )
    if _TOTAL_ROW not in by_category:
        raise DataIntegrityError(
            f"participant file for {session} has no {_TOTAL_ROW} row, so the identity check that "
            "the categories sum correctly cannot be performed"
        )

    measures = [f for f in fields[1:] if f]
    return pl.DataFrame(
        {
            "session_date": [session] * (len(PARTICIPANT_CATEGORIES) * len(measures)),
            "category": [c for c in PARTICIPANT_CATEGORIES for _ in measures],
            "measure": [m for _ in PARTICIPANT_CATEGORIES for m in measures],
            "contracts": [
                _to_float(by_category[c].get(m), session, c, m)
                for c in PARTICIPANT_CATEGORIES
                for m in measures
            ],
        }
    )


def _to_float(raw: str | None, session: date, category: str, measure: str) -> float:
    """Parse a contract count, raising rather than defaulting a blank cell to zero.

    A blank in this file means NSE did not report the cell, which is not the same fact as "no
    contracts traded" — and the two are indistinguishable once a zero has been written.
    """
    if raw is None or raw == "":
        raise DataIntegrityError(
            f"participant file for {session} has an empty cell at {category}/{measure}; refusing "
            "to read it as zero"
        )
    return float(raw.replace(",", ""))


def fetch_participant_session(raw_root: Path, session: date, *, kind: str = "vol",
                              http: requests.Session | None = None) -> pl.DataFrame:
    """Download, validate and immutably cache one session's participant file."""
    target = participant_path(raw_root, session, kind)
    if target.exists():
        return read_parquet(target)
    url = (PARTICIPANT_VOLUME_URL if kind == "vol" else PARTICIPANT_OI_URL).format(
        stamp=f"{session:%d%m%Y}"
    )
    client = http or _new_session()
    response = client.get(url, timeout=60)
    if response.status_code != 200:
        raise DataIntegrityError(f"participant fetch for {session} returned HTTP "
                                 f"{response.status_code} ({url})")
    frame = parse_participant_csv(response.text, session)
    write_raw_parquet(
        frame, target,
        source_url=url,
        extra={"session_date": session.isoformat(), "kind": kind,
               "units": "contracts, NOT rupees; lot sizes vary across the window"},
    )
    return frame


@dataclass(frozen=True)
class FlowStateRule:
    """The parameters defining a flow state. Every one of them is a researcher degree of freedom.

    ``window`` sessions of net activity are summed, standardised against a trailing distribution of
    ``baseline`` sessions, and the state is set by ``threshold`` in standard deviations. Defaults
    are proposals put to the PI at Checkpoint 3.0, not settled choices: a threshold picked after
    seeing which one produced an interesting capacity difference would be precisely the pathology
    this lab exists to detect.
    """

    window: int = 5
    baseline: int = 252
    threshold: float = 1.0


def label_flow_state(net: pl.DataFrame, rule: FlowStateRule) -> pl.DataFrame:
    """Label each session ``inflow`` / ``outflow`` / ``neutral`` from a daily net-activity series.

    The standardising window ends on the previous session, so a state is knowable on the day it
    describes. That is not merely tidy: a flow state fitted with its own future would make every
    downstream capacity comparison conditioned on information the trader did not have, which is the
    leakage class P1's static auditor exists to catch.
    """
    required = {"session_date", "net"}
    missing = required - set(net.columns)
    if missing:
        raise DataIntegrityError(f"net-flow frame is missing columns {sorted(missing)}")
    return (
        net.sort("session_date")
        .with_columns(rolling=pl.col("net").rolling_sum(window_size=rule.window))
        .with_columns(
            baseline_mean=pl.col("rolling").shift(1).rolling_mean(window_size=rule.baseline),
            baseline_sd=pl.col("rolling").shift(1).rolling_std(window_size=rule.baseline),
        )
        .with_columns(
            z=pl.when(pl.col("baseline_sd") > 0)
            .then((pl.col("rolling") - pl.col("baseline_mean")) / pl.col("baseline_sd"))
            .otherwise(None)
        )
        .with_columns(
            flow_state=pl.when(pl.col("z").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("z") >= rule.threshold)
            .then(pl.lit("inflow"))
            .when(pl.col("z") <= -rule.threshold)
            .then(pl.lit("outflow"))
            .otherwise(pl.lit("neutral"))
        )
    )


def net_position_series(participants: pl.DataFrame, category: str = "FII") -> pl.DataFrame:
    """Net long-minus-short contracts per session for one participant category.

    Uses the two total columns rather than summing the individual legs, because the legs are not
    disjoint across the layout revisions in this window and a sum over them double-counts.
    """
    long_col = "Total Long Contracts"
    short_col = "Total Short Contracts"
    wanted = participants.filter(
        (pl.col("category") == category) & pl.col("measure").is_in([long_col, short_col])
    )
    if wanted.height == 0:
        raise DataIntegrityError(
            f"no rows for category {category!r} with the total columns; the file layout has "
            "changed and net position cannot be computed from it"
        )
    return (
        wanted.pivot(on="measure", index="session_date", values="contracts")
        .with_columns(net=pl.col(long_col) - pl.col(short_col))
        .select(["session_date", "net"])
        .sort("session_date")
    )
