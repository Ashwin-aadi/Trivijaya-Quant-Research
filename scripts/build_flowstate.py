"""Build P3's headline artefacts: alpha decay curves and constraint-based deployment capacity.

Answers the two research questions that survived Phase 3.0's measurement, under the PI's ruling of
2026-08-03:

* **RQ2, unchanged** — how fast does each factor's edge decay with holding horizon?
* **RQ1, redefined** — at what deployed AUM can each factor no longer trade inside its own
  participation limit? A *constraint* figure, never an impact-erosion figure.

Plus the flow-conditional comparison, which asks whether deployable size collapses when foreign
participation turns. Writes ``data/processed/flowstate.json`` and a run manifest.

Usage:
    python scripts/build_flowstate.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Any

import polars as pl

from src.capacity.decay import decay_curve, forward_returns, half_life
from src.capacity.deployability import (
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.factors import FACTOR_NAMES, build_factors, daily_universe, long_short_weights
from src.capacity.impact import add_daily_measures
from src.common.config import Config, load_config
from src.common.io import read_parquet, write_derived_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest

_log = get_logger(__name__)

#: Holding horizons in sessions: one day to one quarter. Chosen to bracket the range over which
#: published equity-factor decay is reported, so the comparison in RQ2 is like-for-like.
HORIZONS = (1, 2, 3, 5, 10, 21, 42, 63)
#: Books are formed every session. An earlier version formed monthly, on the reasoning that the
#: signals use trailing windows of a month or more and barely move between sessions. That reasoning
#: was wrong: it confuses how smooth the *signal* is with how precisely its mean return is
#: *estimated*. Monthly formation left 59 observations, every decay curve was visibly ragged, and no
#: t-statistic reached 1.5. Overlapping observations still carry information about the mean, and the
#: overlap is corrected for in the standard error rather than avoided by discarding data.
FORMATION_EVERY = 1


def _panel(cfg: Config) -> pl.DataFrame:
    panel = read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    return panel.filter(
        (pl.col("session_date") >= cfg.dates.dev_start)
        & (pl.col("session_date") <= cfg.dates.dev_end)
    )


def main() -> int:
    cfg = load_config()
    with RunManifest(cfg, "build_flowstate") as run:
        for name in ("prices_adjusted.parquet", "universe.parquet", "participant_flows.parquet"):
            run.add_input(cfg.paths.data_processed / name)

        panel = _panel(cfg)
        universe = read_parquet(cfg.paths.data_processed / "universe.parquet")
        sessions = panel["session_date"].unique().sort().to_list()
        members = daily_universe(universe, sessions)

        # Restrict the panel to point-in-time universe membership BEFORE building signals, so a
        # factor is never ranked against a name the strategy could not have held that day.
        investable = panel.join(members, on=["session_date", "symbol"], how="inner")
        _log.info("investable panel: %d symbol-days, %d symbols",
                  investable.height, investable["symbol"].n_unique())

        scores = build_factors(investable)
        formation = sessions[::FORMATION_EVERY]
        weights = long_short_weights(scores.filter(pl.col("session_date").is_in(formation)))
        books = weights.select(["factor", "session_date"]).unique().height
        _log.info("books: %d factor-sessions", books)

        # --- RQ2: decay -------------------------------------------------------------------
        forward = forward_returns(panel, HORIZONS)
        points = decay_curve(weights, forward, horizons=HORIZONS,
                             formation_spacing=FORMATION_EVERY)
        decay: dict[str, Any] = {}
        for name in FACTOR_NAMES:
            rows = [p for p in points if p.factor == name]
            decay[name] = {
                "curve": [
                    {**asdict(p), "t_statistic": p.t_statistic}
                    for p in sorted(rows, key=lambda p: p.horizon)
                ],
                "half_life_sessions": half_life(rows),
            }

        # --- RQ1: constraint-based capacity ----------------------------------------------
        measured = add_daily_measures(panel, adv_window=cfg.constraints.adv_window_sessions)
        liquidity = measured.select(["session_date", "symbol", "adv_inr"])
        traded = turnover_by_session(
            weights, min_traded_fraction=cfg.constraints.min_traded_fraction
        )
        per_session = session_capacity(
            traded, liquidity, participation_limit=cfg.constraints.max_participation_rate
        )
        summaries = summarise_capacity(
            per_session, participation_limit=cfg.constraints.max_participation_rate
        )

        # --- flow-conditional -------------------------------------------------------------
        flows = read_parquet(cfg.paths.data_processed / "participant_flows.parquet")
        by_state = capacity_by_flow_state(per_session, flows)

        write_derived_parquet(per_session, cfg.paths.data_processed / "capacity_by_session.parquet")
        result = {
            "n_symbol_days_investable": investable.height,
            "n_symbols": investable["symbol"].n_unique(),
            "formation_every_sessions": FORMATION_EVERY,
            "n_formation_dates": len(formation),
            "horizons": list(HORIZONS),
            "participation_limit": cfg.constraints.max_participation_rate,
            "factors_built": list(FACTOR_NAMES),
            "factors_not_built": {
                "value": "requires fundamentals (book value, earnings); budget is zero",
                "quality": "requires fundamentals (ROE, accruals); budget is zero",
            },
            "decay": decay,
            "capacity": [asdict(s) for s in summaries],
            "capacity_by_flow_state": by_state.to_dicts(),
        }
        out = cfg.paths.data_processed / "flowstate.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        run.note("output", str(out))
        _log.info("wrote %s", out)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
