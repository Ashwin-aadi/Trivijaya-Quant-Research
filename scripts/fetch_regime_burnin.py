"""Fetch pre-development index history used solely as HMM fitting burn-in (Phase 2.0, option A).

Why this file exists at all: the calendar series starts 2019-01-01, one year before the development
window. An HMM fit on 2019 alone has never observed a crisis, and would be asked to label the COVID
crash roughly fifty sessions into the labelled period. Expanding-window fitting makes that a real
constraint rather than a nuisance, because there is no legitimate way to borrow the information
from later.

So the burn-in is extended backwards, not forwards. PI-approved 2026-08-01 (DECISIONS.md, Phase 2.0
decision 3), on these terms, which this script does not have the authority to relax:

* The development window, the holdout, and every strategy evaluation are UNCHANGED.
* Pre-2020 sessions initialise the HMM and enter no reported performance metric.
* Nothing here touches, reads, or extends the holdout.

The fetch reuses :func:`fetch_index_sessions` so the burn-in file gets the same immutable-write and
`.meta.json` provenance treatment as every other raw artefact, rather than a bespoke path that
would be exempt from the Phase 1.0 discipline.
"""

from __future__ import annotations

import sys
from datetime import date

from src.common.config import load_config
from src.common.log import get_logger
from src.data.calendar import fetch_index_sessions

_log = get_logger(__name__)

#: Start of the burn-in. Chosen to span the 2015-16 global selloff, demonetisation (Nov 2016) and
#: the Feb 2018 volatility spike, so the model has observed stress before it is asked to name it.
#: Fixed here rather than in config.yaml because it is a one-off provenance fact about a raw file,
#: not a tunable: changing it changes which bytes are in data/raw/, which is write-once.
BURNIN_START = date(2015, 1, 1)


def main() -> int:
    cfg = load_config()
    # The burn-in ends where the calendar begins, so the two files abut without overlapping and
    # neither can silently shadow the other. yfinance treats `end` as exclusive.
    burnin_end = date.fromisoformat(str(cfg.raw["calendar"]["history_start"]))
    if burnin_end <= BURNIN_START:
        _log.error("burn-in start %s is not before calendar start %s", BURNIN_START, burnin_end)
        return 1

    path = cfg.paths.data_raw / "calendar_cnx100_burnin.parquet"
    symbol = cfg.raw["calendar"]["index_symbol"]
    frame = fetch_index_sessions(symbol, BURNIN_START, burnin_end, path)

    first, last = frame["session_date"].min(), frame["session_date"].max()
    _log.info("burn-in: %d sessions, %s .. %s", frame.height, first, last)

    # A loud failure rather than a quiet one: if the fetch silently returned the development period
    # instead of the burn-in, every regime label downstream would be fit on data it must not see.
    if last is not None and last >= burnin_end:  # type: ignore[operator]
        _log.error("burn-in contains a session on or after %s; refusing it", burnin_end)
        return 1

    print(f"burn-in sessions: {frame.height}")
    print(f"range: {first} .. {last}")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
