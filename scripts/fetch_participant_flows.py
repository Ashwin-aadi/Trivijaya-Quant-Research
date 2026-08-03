"""Fetch participant-wise equity-derivatives activity for every development session.

This is the substitute for the FII/DII cash-market net-flow series the charter asks for, which no
free source serves historically. :mod:`src.capacity.flows` documents that finding in full and the
substitution is put to the PI at Checkpoint 3.0; this script is what makes the substitute
measurable, not an acceptance of it.

Writes one immutable raw parquet per session under ``data/raw/participants/``, then a derived
``participant_flows.parquet`` holding the FII net-position series with flow states attached, and a
``flow_diagnostics.json`` reporting coverage and state occupancy. A session NSE does not serve is
recorded as a gap and reported; it is never filled, interpolated, or passed over silently.

Usage:
    python scripts/fetch_participant_flows.py
"""

from __future__ import annotations

import json
import sys
from datetime import date

import polars as pl

from src.capacity.flows import (
    FlowStateRule,
    _new_session,
    fetch_participant_session,
    label_flow_state,
    net_position_series,
)
from src.common.config import Config, load_config
from src.common.exceptions import DataIntegrityError
from src.common.io import read_parquet, write_derived_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest

_log = get_logger(__name__)


def _dev_sessions(cfg: Config) -> list[date]:
    """Trading sessions in the development window, from the Phase 1.0 calendar.

    The holdout calendar is deliberately not consulted. Nothing in Phase 3.0 opens it.
    """
    calendar = read_parquet(cfg.paths.data_raw / "calendar_cnx100.parquet")
    return (
        calendar.filter(
            (pl.col("session_date") >= cfg.dates.dev_start)
            & (pl.col("session_date") <= cfg.dates.dev_end)
        )["session_date"]
        .sort()
        .to_list()
    )


def main() -> int:
    cfg = load_config()
    with RunManifest(cfg, "fetch_participant_flows") as run:
        sessions = _dev_sessions(cfg)
        _log.info("fetching participant activity for %d sessions", len(sessions))

        http = _new_session()
        frames: list[pl.DataFrame] = []
        gaps: list[str] = []
        for index, session in enumerate(sessions):
            try:
                frames.append(fetch_participant_session(cfg.paths.data_raw, session, http=http))
            except (DataIntegrityError, OSError) as exc:
                gaps.append(f"{session}: {type(exc).__name__}")
            if index and index % 200 == 0:
                _log.info("  %d/%d sessions, %d gaps so far", index, len(sessions), len(gaps))

        if not frames:
            raise DataIntegrityError("no participant session was fetched; refusing to continue")
        participants = pl.concat(frames)

        net = net_position_series(participants, category="FII")
        rule = FlowStateRule()
        labelled = label_flow_state(net, rule)
        out = cfg.paths.data_processed / "participant_flows.parquet"
        write_derived_parquet(labelled, out)

        occupancy = (
            labelled.drop_nulls("flow_state")
            .group_by("flow_state")
            .agg(n=pl.len())
            .sort("flow_state")
        )
        diagnostics = {
            "sessions_requested": len(sessions),
            "sessions_fetched": len(frames),
            "gap_count": len(gaps),
            "gaps": gaps[:50],
            "net_series_sessions": labelled.height,
            "labelled_sessions": int(labelled["flow_state"].is_not_null().sum()),
            "unlabelled_burn_in_sessions": int(labelled["flow_state"].is_null().sum()),
            "flow_state_rule": {
                "window": rule.window, "baseline": rule.baseline, "threshold": rule.threshold,
            },
            "occupancy": dict(zip(occupancy["flow_state"].to_list(),
                                  occupancy["n"].to_list(), strict=True)),
            "units": "contracts, NOT rupees; see src/capacity/flows.py",
        }
        path = cfg.paths.data_processed / "flow_diagnostics.json"
        path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
        run.note("sessions_fetched", len(frames))
        run.note("gap_count", len(gaps))
        _log.info("wrote %s and %s", out, path)
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
