"""Download and validate the daily bhavcopy for every trading session in the development window.

Resumable: sessions already cached are skipped, so an interruption costs one session rather than
the whole run. Every file is content-validated on the way in (see src/data/bhavcopy.py) — a date
NSE will not serve raises instead of silently caching a web page.

Deliberately stops at the end of the development window. Holdout data is not downloaded here;
fetching it needs its own PI authorization so the holdout stays genuinely untouched.

Usage:
    python scripts/download_bhavcopy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the repo root importable when this file is run directly as a script, before any `src`
# import is attempted. Must precede those imports, hence the E402 suppression below.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config  # noqa: E402
from src.common.exceptions import DataIntegrityError  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.data.bhavcopy import fetch_session, session_path  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402

_log = get_logger("download_bhavcopy")

# A handful of sessions can legitimately be unavailable upstream. Abort if failures pile up, since
# that means the source is broken rather than a date being genuinely missing.
MAX_CONSECUTIVE_FAILURES = 12


def main() -> int:
    cfg = load_config()
    calendar_file = cfg.paths.data_raw / "calendar_cnx100.parquet"
    calendar = load_calendar(calendar_file)
    sessions = calendar.sessions_in_range(cfg.dates.dev_start, cfg.dates.dev_end)

    pending = [s for s in sessions if not session_path(cfg.paths.data_raw, s).exists()]
    _log.info("dev window %s..%s: %d sessions, %d still to download",
              cfg.dates.dev_start, cfg.dates.dev_end, len(sessions), len(pending))

    failures: list[tuple[str, str]] = []
    consecutive = 0
    started = time.time()

    with RunManifest(cfg, script="scripts/download_bhavcopy.py") as run:
        run.add_input(calendar_file)
        run.note("sessions_in_window", len(sessions))
        for index, session in enumerate(pending, start=1):
            try:
                fetch_session(cfg.paths.data_raw, session)
                consecutive = 0
            except (DataIntegrityError, OSError, ValueError) as exc:
                # Record and continue: one unavailable date should not discard the whole run, but
                # every failure is reported at the end rather than swallowed.
                failures.append((session.isoformat(), f"{type(exc).__name__}: {exc}"))
                consecutive += 1
                _log.warning("session %s failed (%d consecutive): %s", session, consecutive, exc)
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    _log.error("aborting: %d consecutive failures suggests the source is broken",
                               consecutive)
                    break
            if index % 50 == 0:
                rate = index / max(time.time() - started, 1e-9)
                _log.info("%d/%d done (%.1f sessions/s, %d failures)",
                          index, len(pending), rate, len(failures))

        cached = sum(1 for s in sessions if session_path(cfg.paths.data_raw, s).exists())
        run.note("sessions_cached", cached)
        run.note("failures", failures)

    _log.info("finished: %d/%d sessions cached, %d failures", cached, len(sessions), len(failures))
    if failures:
        _log.warning("first failures: %s", failures[:5])
    # Non-zero exit if coverage is materially incomplete, so a shortfall cannot pass unnoticed.
    return 0 if cached >= 0.98 * len(sessions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
