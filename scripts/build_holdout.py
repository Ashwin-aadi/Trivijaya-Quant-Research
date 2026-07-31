"""Acquire and build the holdout-period data, once, under explicit PI authorisation.

The holdout was never downloaded during development. That was deliberate: Rule 7 makes it sacred,
and data that does not exist on disk cannot be accidentally read by a stray script, a notebook, or
a wildcard glob. The cost of that safety is this script — the holdout has to be *acquired* before
it can be evaluated, and acquisition is itself a privileged act.

**The window is read from config, never from what happens to be available.** `holdout_end` was
frozen before this script was first run. A holdout whose end moves with the newest session is not a
fixed window; its length becomes a free parameter, and choosing it after seeing a result is exactly
the pathology this lab exists to detect.

**Nothing here overwrites development artifacts.** Outputs carry a `_holdout` suffix and the
development panel, universe and calendar are left byte-identical, so the development results stay
reproducible after the holdout exists.

**The universe stays survivorship-free.** It is derived from trailing median traded value, not
looked up from a membership list, so it extends into the holdout by the same rule that built it for
development — no list that knows the future is ever consulted.

Usage:
    python scripts/build_holdout.py --authorised-by "PI, 2026-07-31, checkpoint 1.4 Q1"
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.config import Config, load_config  # noqa: E402
from src.common.io import write_derived_parquet  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.data.bhavcopy import fetch_session, session_path  # noqa: E402
from src.data.calendar import (  # noqa: E402
    TradingCalendar,
    fetch_index_sessions,
    load_calendar,
)
from src.data.corporate_actions import (  # noqa: E402
    apply_adjustments,
    fetch_declared_splits,
    reconcile_with_prices,
    resolve_disputed,
)
from src.data.prices import load_panel  # noqa: E402
from src.data.universe import build_universe_history, snapshots_to_frame  # noqa: E402

_log = get_logger("build_holdout")

MAX_CONSECUTIVE_FAILURES = 12

#: Pause between requests. The first attempt ran at 1.4 sessions/s and NSE began timing out after
#: about 25 files, so this is deliberately unhurried — the whole window is a few hundred sessions
#: and there is nothing to gain by hammering a public archive.
THROTTLE_SECONDS = 1.5

#: Attempts per session before recording it as failed. Timeouts here are transient.
MAX_ATTEMPTS = 3


def _fetch_with_retry(raw_root: Path, session: date) -> str | None:
    """Fetch one session, retrying on transient failure. Returns an error string, None on success.

    Catches ``Exception`` deliberately. ``jugaad_data.full_bhavcopy_raw`` swallows ``ReadTimeout``
    for dates from 2020 on and then falls through to ``return r.text`` with ``r`` unbound, so a
    plain network timeout arrives as ``UnboundLocalError``. Enumerating exception types against a
    third-party library that raises whatever its own bugs produce is not a workable contract; the
    failure is recorded either way, and nothing is silently swallowed.
    """
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            fetch_session(raw_root, session)
            return None
        except Exception as exc:  # noqa: BLE001 - see docstring
            last = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < MAX_ATTEMPTS:
                # Back off before retrying: the failure mode observed is a timeout under load, and
                # retrying immediately reproduces it.
                time.sleep(2.0 * attempt)
    return last


def holdout_calendar(cfg: Config) -> TradingCalendar:
    """Sessions spanning development *and* holdout, fetched once and cached.

    The development calendar stops at `dev_end`, so it cannot answer questions about 2025. A second
    file is fetched rather than the first being extended: `data/raw/` is write-once, and rewriting a
    file the development results were built from would break their reproducibility.
    """
    path = cfg.paths.data_raw / "calendar_cnx100_holdout.parquet"
    if not path.exists():
        _log.info("fetching index sessions through %s", cfg.dates.holdout_end)
        fetch_index_sessions(
            symbol=cfg.calendar.index_symbol,
            start=cfg.calendar.history_start,
            end=cfg.dates.holdout_end,
            raw_path=path,
        )
    calendar = load_calendar(path)
    if calendar.last_session < cfg.dates.holdout_end:
        _log.warning(
            "calendar ends %s, before the frozen holdout_end %s. The window is NOT shortened to "
            "match — that would silently redefine the holdout. Sessions actually present are used "
            "and the shortfall is reported.",
            calendar.last_session, cfg.dates.holdout_end,
        )
    _log.info("holdout calendar: %s -> %s (%d sessions)",
              calendar.first_session, calendar.last_session, calendar.n_sessions)
    return calendar


def download(cfg: Config, calendar: TradingCalendar) -> list[date]:
    """Fetch every bhavcopy in the holdout window. Resumable: cached sessions are skipped."""
    sessions = calendar.sessions_in_range(cfg.dates.holdout_start, cfg.dates.holdout_end)
    pending = [s for s in sessions if not session_path(cfg.paths.data_raw, s).exists()]
    _log.info("holdout window %s..%s: %d sessions, %d to download",
              cfg.dates.holdout_start, cfg.dates.holdout_end, len(sessions), len(pending))

    failures: list[tuple[str, str]] = []
    consecutive = 0
    started = time.time()
    for index, session in enumerate(pending, start=1):
        error = _fetch_with_retry(cfg.paths.data_raw, session)
        if error is None:
            consecutive = 0
        else:
            failures.append((session.isoformat(), error))
            consecutive += 1
            _log.warning("session %s failed after %d attempts (%d consecutive): %s",
                         session, MAX_ATTEMPTS, consecutive, error)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                _log.error("aborting: %d consecutive failures means the source is broken",
                           consecutive)
                break
        time.sleep(THROTTLE_SECONDS)
        if index % 25 == 0:
            rate = index / max(time.time() - started, 1e-9)
            _log.info("%d/%d fetched (%.1f/s, %d failures)",
                      index, len(pending), rate, len(failures))

    if failures:
        _log.warning("%d sessions could not be fetched; they are reported, not silently dropped",
                     len(failures))
        for day, reason in failures[:10]:
            _log.warning("  %s: %s", day, reason)
    return [s for s in sessions if session_path(cfg.paths.data_raw, s).exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorised-by", type=str, default=None,
                        help="the PI's explicit authorisation, recorded in the output")
    args = parser.parse_args()

    if not args.authorised_by:
        _log.error(
            "the holdout may be acquired and evaluated once per project, and only with explicit "
            "PI authorisation (Rule 7). Pass --authorised-by with the authorisation text."
        )
        return 2

    cfg = load_config()
    seed_everything(cfg.meta.seed)
    _log.warning("HOLDOUT ACQUISITION, authorised by: %s", args.authorised_by)

    calendar = holdout_calendar(cfg)
    available = download(cfg, calendar)
    if not available:
        _log.error("no holdout sessions available; nothing to build")
        return 1

    # The panel reaches back a trailing window before the holdout starts. Strategies need lookback
    # history on their first holdout session, and the universe ranking needs the same. That history
    # is development data, already on disk, and using it is not a holdout violation — it is what a
    # contemporaneous investor standing on 2025-01-01 would have had.
    warmup_start = calendar.shift(available[0], -cfg.universe.trailing_sessions)
    panel_sessions = calendar.sessions_in_range(warmup_start, available[-1])
    cached = [s for s in panel_sessions if session_path(cfg.paths.data_raw, s).exists()]
    _log.info("panel window %s..%s: %d sessions, %d cached",
              warmup_start, available[-1], len(panel_sessions), len(cached))

    panel = load_panel(cfg.paths.data_raw, cached)
    _log.info("panel: %d rows, %d symbols, %s -> %s", panel.height, panel["symbol"].n_unique(),
              panel["session_date"].min(), panel["session_date"].max())

    # Universe first, and deliberately before any corporate-action adjustment. Ranking uses the
    # exchange's own reported rupee turnover, which is already invariant to capital changes — a
    # split moves price and share count in opposite directions — so the universe stays independent
    # of adjustment and immune to any error in it. Same rule, same parameters, same code path as
    # development; only the window differs.
    unadjusted = apply_adjustments(panel, [])
    snapshots = build_universe_history(
        unadjusted, calendar, cfg.dates.holdout_start, available[-1], cfg.universe
    )
    universe_frame = snapshots_to_frame(snapshots)
    write_derived_parquet(universe_frame, cfg.paths.data_processed / "universe_holdout.parquet")
    _log.info("universe: %d rebalances, %d rows", len(snapshots), universe_frame.height)

    # Then adjustment, over exactly the universe names, exactly as development did it. Skipping
    # this would leave raw prices in which every split reads as a large one-day loss, and a
    # strategy's holdout return would be manufactured by the data rather than by the strategy.
    symbols = sorted(universe_frame["symbol"].unique().to_list())
    _log.info("fetching declared corporate actions for %d holdout universe symbols", len(symbols))
    declared, failed = [], []
    for index, symbol in enumerate(symbols, start=1):
        try:
            declared.extend(
                fetch_declared_splits(symbol, cfg.dates.holdout_start, available[-1])
            )
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not abort the run
            failed.append(f"{symbol}: {type(exc).__name__}")
        if index % 25 == 0:
            _log.info("%d/%d symbols fetched", index, len(symbols))
    if failed:
        _log.warning("%d symbols failed corporate-action lookup: %s", len(failed), failed[:8])

    agreed, disputed = reconcile_with_prices(declared, unadjusted, calendar)
    snapped, decisions = resolve_disputed(disputed)
    _log.info("corporate actions: %d declared, %d agreed, %d disputed, %d snapped",
              len(declared), len(agreed), len(disputed), len(snapped))

    base = panel.drop([c for c in panel.columns if c.startswith("adj_") or c == "divisor"])
    adjusted = apply_adjustments(base, agreed + snapped)
    out = cfg.paths.data_processed / "prices_adjusted_holdout.parquet"
    write_derived_parquet(adjusted, out)

    print(f"holdout sessions available: {len(available)}  (frozen end {cfg.dates.holdout_end})")
    print(f"window: {panel['session_date'].min()} -> {panel['session_date'].max()}")
    print(f"panel rows: {adjusted.height}   symbols: {adjusted['symbol'].n_unique()}")
    print(f"universe rebalances: {len(snapshots)}")
    print(f"corporate actions applied: {len(agreed) + len(snapped)}"
          f"  (disputed {len(disputed)}, resolved {len(decisions)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
