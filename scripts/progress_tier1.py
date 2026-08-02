"""Show how far the Tier 1 stress run has got, and whether it is still alive.

Reads state from disk rather than holding a handle on the run, so it is safe to call at any time,
tells the truth after a crash or a reboot, and writes nothing.

**Liveness is the point.** A run that is merely slow and a run that has hung look identical in a
completion count, so the headline here is *seconds since the last backtest finished*, not the
percentage. Paths take roughly 50 minutes each and land in bursts as workers finish together, so
the completion count can sit still for a long time while everything is fine; the heartbeat cannot.

Usage:
    python scripts/progress_tier1.py
    python scripts/progress_tier1.py --watch      # refresh every 30 seconds
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("runs/tier1")
LOG = OUT / "run.log"
TOTAL_PATHS = 100
STRATEGIES = 158
#: Beyond this with no completed backtest, something is wrong. The slowest single backtest measured
#: during calibration was 166s; three times that is generous rather than tight.
STALL_SECONDS = 500


def _log_heartbeat() -> tuple[int, float | None, str]:
    """Backtests finished, seconds since the most recent one, and its description."""
    if not LOG.exists():
        return 0, None, "no log yet"
    done = 0
    newest: datetime | None = None
    detail = ""
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if "final equity" not in line and "ruined on" not in line:
            continue
        done += 1
        try:
            stamp = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
            newest = stamp.replace(tzinfo=UTC)
            detail = line.split("|")[-1].strip()[:60]
        except ValueError:
            continue
    if newest is None:
        return done, None, detail
    return done, (datetime.now(UTC) - newest).total_seconds(), detail


def _completed_paths() -> list[dict[str, object]]:
    """One record per finished path, read from the per-path files rather than the progress log."""
    records = []
    for path_file in sorted(OUT.glob("path_*.json")):
        try:
            records.append(json.loads(path_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue          # a file being written right now; it will be there next refresh
    return records


def report() -> None:
    paths = _completed_paths()
    backtests, since, detail = _log_heartbeat()
    expected = TOTAL_PATHS * STRATEGIES

    print(f"\n  Tier 1 — {STRATEGIES} strategies x {TOTAL_PATHS} counterfactual paths")
    print(f"  {'-' * 66}")
    print(f"  paths complete     {len(paths):>4} / {TOTAL_PATHS}"
          f"      ({len(paths) / TOTAL_PATHS:.0%})")
    print(f"  backtests done     {backtests:>4} / {expected}"
          f"   ({backtests / expected:.1%})")

    if since is None:
        print("  heartbeat          nothing finished yet — normal for the first minute")
    else:
        state = "RUNNING" if since < STALL_SECONDS else "*** POSSIBLY STUCK ***"
        print(f"  last backtest      {since:>4.0f}s ago      {state}")
        print(f"  most recent        {detail}")

    if paths:
        seconds = [float(p["seconds"]) for p in paths]  # type: ignore[arg-type]
        per_path = sum(seconds) / len(seconds)
        # 24 workers run concurrently, so wall-clock per path is far below the in-worker duration.
        elapsed = _elapsed()
        # Estimated from backtests, not paths. Paths complete in bursts of ~24 as workers finish
        # together, so a path-based rate is measured off a staircase and reads as *rising* time
        # remaining between bursts. The backtest count is the underlying ramp and is monotone.
        share = backtests / expected if expected else 0.0
        left = (elapsed / share - elapsed) / 60 if share else float("nan")
        print(f"  mean path duration {per_path / 60:>4.0f} min in-worker")
        print(f"  elapsed            {elapsed / 60:>4.0f} min")
        print(f"  estimated left     {left:>4.0f} min")
        missing = [float(p["diagnostics"]["missing_member_rate"]) for p in paths]  # type: ignore[index,arg-type]
        worst = max(float(p["diagnostics"]["max_abs_session_return"]) for p in paths)  # type: ignore[index,arg-type]
        print(f"  missing members    {sum(missing) / len(missing):.2%} mean"
              f"   (real panel baseline 0.68%)")
        # 126.2% is SUZLON on 2021-02-02, across an 85-session suspension. The builder resamples
        # returns between consecutive *available* closes, so gap-spanning moves are real and
        # expected; 76.9% is the ceiling for calendar-adjacent sessions only and is the wrong
        # comparison. Matching 126.2% is evidence the reconstruction is faithful, not a warning.
        print(f"  largest 1-day move {worst:.1%}"
              f"       (real panel maximum 126.2% — above it means a bug)")
        evaluated = [int(p.get("results") and sum(  # type: ignore[arg-type]
            1 for r in p["results"] if r["outcome"] == "evaluated")) for p in paths]  # type: ignore[union-attr]
        if min(evaluated) < STRATEGIES:
            print(f"  ** {STRATEGIES - min(evaluated)} strategies failed on at least one path **")
    print()


def _elapsed() -> float:
    """Seconds since the run started, from the log's first line."""
    if not LOG.exists():
        return 0.0
    first = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if not first:
        return 0.0
    try:
        started = datetime.strptime(first[0][:23], "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return 0.0
    return (datetime.now(UTC) - started.replace(tzinfo=UTC)).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="refresh every 30 seconds")
    args = parser.parse_args()
    while True:
        report()
        if not args.watch:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
