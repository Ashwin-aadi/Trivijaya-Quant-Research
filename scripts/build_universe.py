"""Build the point-in-time universe and adjusted price panel from cached bhavcopy sessions.

Order matters and is enforced by the code rather than by documentation:
  1. Load the trading calendar (defines which sessions must exist).
  2. Load every cached session into one panel, failing if any is absent.
  3. Assert the panel is gap-free before anything reads across sessions.
  4. Select the universe at each rebalance, using only sessions strictly before it.

Run `build_corporate_actions.py` next: it resolves splits from declared corporate actions and
rewrites the adjusted price columns. The universe itself ranks on the exchange's reported rupee
turnover, which no capital change can distort, so it does not depend on that step.

Writes the universe history and the price panel to data/processed/, with a run manifest.

Usage:
    python scripts/build_universe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.common.config import load_config  # noqa: E402
from src.common.io import write_derived_parquet  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.data.bhavcopy import session_path  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.data.corporate_actions import (  # noqa: E402
    apply_adjustments,
    assert_panel_contiguous,
)
from src.data.prices import load_panel  # noqa: E402
from src.data.universe import build_universe_history, snapshots_to_frame  # noqa: E402

_log = get_logger("build_universe")


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.meta.seed)
    calendar_file = cfg.paths.data_raw / "calendar_cnx100.parquet"
    calendar = load_calendar(calendar_file)

    # The trailing window reaches back before dev_start, so the panel must start earlier than the
    # first rebalance or the very first universe would be built on truncated history.
    panel_start = calendar.shift(
        calendar.next_session(cfg.dates.dev_start, strictly_after=False),
        -cfg.universe.trailing_sessions,
    )
    sessions = calendar.sessions_in_range(panel_start, cfg.dates.dev_end)
    cached = [s for s in sessions if session_path(cfg.paths.data_raw, s).exists()]
    _log.info("panel window %s..%s: %d sessions, %d cached",
              panel_start, cfg.dates.dev_end, len(sessions), len(cached))

    with RunManifest(cfg, script="scripts/build_universe.py") as run:
        run.add_input(calendar_file)
        panel = load_panel(cfg.paths.data_raw, sessions)
        run.note("panel_rows", panel.height)
        run.note("panel_symbols", panel["symbol"].n_unique())

        # Confirm the panel has no holes before anything reads across sessions.
        assert_panel_contiguous(panel, calendar)

        # No adjustment is applied here. Splits are resolved from declared corporate actions in
        # build_corporate_actions.py, which runs next; inferring them from the exchange's restated
        # previous close was found to miss roughly 40% of real events. The pass-through keeps the
        # adj_* columns present so the schema is stable for whatever reads this first.
        adjusted = apply_adjustments(panel, [])

        # Rank on the exchange's own reported rupee turnover rather than reconstructing it from
        # adjusted price times adjusted volume. Turnover is already invariant to splits — a
        # capital change moves price and share count in opposite directions — so using it keeps
        # the universe entirely independent of corporate-action adjustment, and therefore immune
        # to any error in it.
        snapshots = build_universe_history(
            adjusted, calendar, cfg.dates.dev_start, cfg.dates.dev_end, cfg.universe
        )
        run.note("rebalances", len(snapshots))

        universe_frame = snapshots_to_frame(snapshots)
        write_derived_parquet(universe_frame, cfg.paths.data_processed / "universe.parquet")
        write_derived_parquet(adjusted, cfg.paths.data_processed / "prices_adjusted.parquet")
        run.note("universe_rows", universe_frame.height)

    _log.info("universe built: %d rebalances over %d sessions", len(snapshots), len(sessions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
