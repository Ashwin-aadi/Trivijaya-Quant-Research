"""Label every development session with an expanding-window HMM regime, and measure the labels.

Produces three artefacts:

* ``regime_labels.parquet`` — the causal labels. One row per labelled session, carrying the refit
  that produced it and that refit's training size. These are the only labels any downstream P2 work
  may use.
* ``regime_diagnostics.json`` — occupancy, the label-stability measurements required by the PI, and
  the dated regime timeline.
* a run manifest, as every script in this repository writes.

**On the stability diagnostic and why it is not a leak.** Measuring how a label revises requires
relabelling past sessions with later-fitted models, which is by construction non-causal. That is
legitimate *for a diagnostic* and illegitimate for a label. The two are kept in separate files with
separate names for exactly that reason: nothing downstream reads the diagnostic, and the labels
file contains only the causal assignment.

Usage:
    python scripts/build_regimes.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date

import numpy as np
import polars as pl

from src.audit.stat import TrialCounter
from src.common.config import load_config
from src.common.exceptions import DataIntegrityError
from src.common.io import write_derived_parquet
from src.common.log import get_logger
from src.common.manifest import RunManifest
from src.stress.inputs import BURNIN_FILE, CALENDAR_FILE, load_index_closes
from src.stress.regimes import (
    FEATURE_NAMES,
    RegimeModel,
    build_features,
    expanding_window_labels,
    feature_matrix,
)

_log = get_logger(__name__)


def _frozen_k(cfg) -> int:  # noqa: ANN001 - Config, imported for typing only in the caller
    """Read the frozen K. Refuses to run if selection has not happened, rather than picking one."""
    path = cfg.paths.data_processed / "regime_k_selection.json"
    if not path.exists():
        raise DataIntegrityError(
            f"{path} is missing; run scripts/select_regime_states.py first. "
            "Choosing K here would be selecting it with sight of the labelling window."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["selected_k"])


def _refit_dates(cfg) -> list[date]:  # noqa: ANN001
    """P1's quarterly universe-rebalance dates, reused rather than a new schedule invented."""
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    return sorted(universe["rebalance_date"].unique().to_list())


def _stability(
    labels: pl.DataFrame,
    models: list[RegimeModel],
    matrix: np.ndarray,
) -> dict[str, object]:
    """Revision rates: how often a session's label changes when refit with more data.

    Two statistics, both split by calendar year because the whole expectation is that early labels
    are less stable and a pooled number would hide it:

    * **one-quarter revision** — label under the model that produced it, versus the next refit.
    * **terminal disagreement** — versus the final refit of the window.

    The observations up to ``t`` are identical between refits; only the parameters differ. Any
    revision is therefore attributable to parameter drift alone.
    """
    per_model = [model.filtered_states(matrix) for model in models]
    train_sizes = [model.n_train for model in models]
    index_of = {size: i for i, size in enumerate(train_sizes)}
    rows = labels.with_row_index("row")

    one_quarter: Counter[int] = Counter()
    one_quarter_total: Counter[int] = Counter()
    terminal: Counter[int] = Counter()
    terminal_total: Counter[int] = Counter()

    dates = rows["session_date"].to_list()
    positions = {day: i for i, day in enumerate(dates)}
    for row in rows.iter_rows(named=True):
        model_index = index_of[row["n_train"]]
        # Position of this session within the full feature matrix, not within the labels frame.
        matrix_position = row["matrix_position"]
        year = row["session_date"].year
        assigned = int(per_model[model_index][matrix_position])

        if model_index + 1 < len(per_model):
            one_quarter_total[year] += 1
            if int(per_model[model_index + 1][matrix_position]) != assigned:
                one_quarter[year] += 1
        terminal_total[year] += 1
        if int(per_model[-1][matrix_position]) != assigned:
            terminal[year] += 1

    def _rates(hits: Counter[int], totals: Counter[int]) -> dict[str, object]:
        by_year = {
            str(year): {"changed": hits[year], "n": totals[year], "rate": hits[year] / totals[year]}
            for year in sorted(totals)
        }
        total_n = sum(totals.values())
        return {
            "by_year": by_year,
            "pooled": {
                "changed": sum(hits.values()),
                "n": total_n,
                "rate": (sum(hits.values()) / total_n) if total_n else None,
            },
        }

    assert len(positions) == labels.height  # every session labelled exactly once
    return {
        "one_quarter_revision": _rates(one_quarter, one_quarter_total),
        "terminal_disagreement": _rates(terminal, terminal_total),
    }


def _timeline(labels: pl.DataFrame) -> list[dict[str, object]]:
    """Contiguous runs of a single state, so a human can read the labelling against memory."""
    runs: list[dict[str, object]] = []
    for row in labels.iter_rows(named=True):
        if runs and runs[-1]["state"] == row["state"]:
            runs[-1]["end"] = str(row["session_date"])
            runs[-1]["sessions"] = int(runs[-1]["sessions"]) + 1  # type: ignore[arg-type]
            continue
        runs.append(
            {
                "state": row["state"],
                "start": str(row["session_date"]),
                "end": str(row["session_date"]),
                "sessions": 1,
            }
        )
    return runs


def main() -> int:
    cfg = load_config()
    with RunManifest(cfg, script="build_regimes.py") as run:
        for name in (BURNIN_FILE, CALENDAR_FILE):
            run.add_input(cfg.paths.data_raw / name)

        k = _frozen_k(cfg)
        dev_start = date.fromisoformat(str(cfg.raw["dates"]["dev_start"]))
        dev_end = date.fromisoformat(str(cfg.raw["dates"]["dev_end"]))
        series = load_index_closes(cfg).filter(pl.col("session_date") <= dev_end)
        features = build_features(series)
        matrix = feature_matrix(features)

        refits = [day for day in _refit_dates(cfg) if day >= dev_start]
        _log.info("K=%d (frozen), %d refit dates, %d feature rows", k, len(refits), matrix.shape[0])

        ledger = TrialCounter(cfg.paths.data_processed / "regime_ledger.jsonl")
        labels, models = expanding_window_labels(features, refits, k=k, seed=cfg.meta.seed)
        ledger.record(f"regime_labelling_k{k}", "evaluated")

        # Attach each label's row in the feature matrix so the diagnostic indexes the same array
        # the models were decoded over, rather than re-deriving positions and risking an off-by-one.
        positions = {day: i for i, day in enumerate(features["session_date"].to_list())}
        labels = labels.with_columns(
            pl.col("session_date").replace_strict(positions, return_dtype=pl.Int64)
            .alias("matrix_position")
        )

        if labels["session_date"].min() > dev_start:
            _log.warning(
                "labels begin %s, after dev_start %s", labels["session_date"].min(), dev_start
            )

        occupancy = (
            labels.group_by("state").len().sort("state").rename({"len": "sessions"})
        )
        diagnostics = {
            "k": k,
            "features": list(FEATURE_NAMES),
            "n_labelled_sessions": labels.height,
            "n_refits": len(models),
            "first_labelled": str(labels["session_date"].min()),
            "last_labelled": str(labels["session_date"].max()),
            "training_sessions_first_refit": models[0].n_train,
            "training_sessions_last_refit": models[-1].n_train,
            "occupancy": {
                str(row["state"]): row["sessions"] for row in occupancy.iter_rows(named=True)
            },
            "state_mean_log_vol_final_refit": models[-1].means[:, 0].tolist(),
            "state_mean_cum_return_final_refit": models[-1].means[:, 1].tolist(),
            "stability": _stability(labels, models, matrix),
            "timeline": _timeline(labels),
            "trials_recorded": ledger.verify(),
        }

        out_labels = cfg.paths.data_processed / "regime_labels.parquet"
        write_derived_parquet(labels.drop("matrix_position"), out_labels)
        out_diag = cfg.paths.data_processed / "regime_diagnostics.json"
        out_diag.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
        run.note("k", k)
        run.note("n_labelled_sessions", labels.height)

    print(f"labelled {labels.height} sessions, K={k}, {len(models)} refits")
    print(f"occupancy: {diagnostics['occupancy']}")
    stability = diagnostics["stability"]
    print(f"one-quarter revision rate (pooled): {stability['one_quarter_revision']['pooled']}")
    print(f"terminal disagreement (pooled):     {stability['terminal_disagreement']['pooled']}")
    print(f"written: {out_labels}\n         {out_diag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
