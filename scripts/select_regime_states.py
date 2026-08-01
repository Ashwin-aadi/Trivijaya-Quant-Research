"""Select the HMM state count K once, by BIC, on the burn-in window only — then freeze it.

Two properties make this defensible, and both are enforced here rather than documented and hoped
for (PI ruling, DECISIONS.md Phase 2.0 decision 5):

1. **Selection sees no evaluation data.** K is chosen on sessions strictly before
   ``dates.dev_start``. No strategy is ever scored on those sessions, so choosing K cannot be
   choosing it to suit a result.
2. **Selection happens once.** The output file is write-once. If it exists, this script refuses to
   overwrite it and exits non-zero. Re-selecting K after seeing labels — or after seeing anything
   downstream of them — is precisely the post-hoc tuning P2 is built to avoid.

BIC is a likelihood-penalty criterion. No backtest runs here and no performance metric is read.

Usage:
    python scripts/select_regime_states.py
"""

from __future__ import annotations

import json
import sys
from datetime import date

from src.audit.stat import TrialCounter
from src.common.config import load_config
from src.common.log import get_logger
from src.common.manifest import RunManifest
from src.stress.inputs import BURNIN_FILE, CALENDAR_FILE, burnin_only, load_index_closes
from src.stress.regimes import (
    FEATURE_NAMES,
    bayesian_information_criterion,
    build_features,
    feature_matrix,
    fit_regime_model,
    n_free_parameters,
)

_log = get_logger(__name__)

#: Candidate state counts, fixed by the charter's "2-4 states" and not widened after seeing BIC.
CANDIDATE_K = (2, 3, 4)

OUTPUT_NAME = "regime_k_selection.json"


def main() -> int:
    cfg = load_config()
    output = cfg.paths.data_processed / OUTPUT_NAME
    if output.exists():
        _log.error(
            "%s already exists. K is frozen; re-selecting it would be post-hoc tuning. "
            "Delete it deliberately and log the reason in DECISIONS.md if that is really intended.",
            output,
        )
        return 2

    with RunManifest(cfg, script="select_regime_states.py") as run:
        for name in (BURNIN_FILE, CALENDAR_FILE):
            run.add_input(cfg.paths.data_raw / name)

        dev_start = date.fromisoformat(str(cfg.raw["dates"]["dev_start"]))
        series = burnin_only(load_index_closes(cfg), dev_start)
        features = build_features(series)
        matrix = feature_matrix(features)
        _log.info(
            "selecting K on %d feature rows, %s .. %s (all strictly before %s)",
            matrix.shape[0],
            features["session_date"].min(),
            features["session_date"].max(),
            dev_start,
        )

        # Every fit attempted is recorded, successes and failures alike, on P2's own ledger.
        # P1's trial_ledger.jsonl is immutable: its N = 1,887 underpins every published DSR.
        ledger = TrialCounter(cfg.paths.data_processed / "regime_ledger.jsonl")

        rows = []
        for k in CANDIDATE_K:
            try:
                model = fit_regime_model(
                    matrix, k=k, fitted_through=features["session_date"].max(), seed=cfg.meta.seed
                )
            except Exception as exc:  # noqa: BLE001 - recorded, re-raised below, never swallowed
                ledger.record(f"regime_k{k}", "runtime_error")
                _log.error("K=%d failed to fit: %s", k, exc)
                raise
            ledger.record(f"regime_k{k}", "evaluated")
            rows.append(
                {
                    "k": k,
                    "log_likelihood": model.log_likelihood,
                    "free_parameters": n_free_parameters(k, matrix.shape[1]),
                    "bic": bayesian_information_criterion(model, matrix),
                    "state_mean_log_vol": model.means[:, 0].tolist(),
                    "state_mean_cum_return": model.means[:, 1].tolist(),
                }
            )
            _log.info("K=%d  logL=%.3f  BIC=%.3f", k, model.log_likelihood, rows[-1]["bic"])

        best = min(rows, key=lambda row: float(row["bic"]))
        payload = {
            "selected_k": best["k"],
            "criterion": "BIC (lower is better)",
            "candidates": rows,
            "selection_window": {
                "start": str(features["session_date"].min()),
                "end": str(features["session_date"].max()),
                "n_feature_rows": int(matrix.shape[0]),
                "dev_start_excluded_from_here": str(dev_start),
            },
            "features": list(FEATURE_NAMES),
            "seed": cfg.meta.seed,
            "trials_recorded": ledger.verify(),
            "frozen": True,
            "note": (
                "K is frozen at this value permanently. Quarterly refits re-estimate parameters "
                "and must never re-select K."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        run.note("selected_k", best["k"])
        run.note("bic_table", rows)

    print(f"\nselected K = {best['k']} by BIC on {matrix.shape[0]} pre-{dev_start} sessions")
    for row in rows:
        marker = "  <-- selected" if row["k"] == best["k"] else ""
        print(f"  K={row['k']}  logL={row['log_likelihood']:12.3f}  BIC={row['bic']:12.3f}{marker}")
    print(f"\nfrozen to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
