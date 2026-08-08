"""Deflate the eleven standard factor strategies, and run the semantic and PBO layers over them.

**Why this exists.** The statistical auditor rejected 100% of every AI corpus it was shown, and a
layer that rejects everything is either correct about a worthless population or non-informative
under this evaluation regime. The two readings are indistinguishable from the corpus alone. The
eleven standard factors are the population that separates them: they are published, decades-old,
and were hand-written as Phase 1.2 fixtures rather than drawn from any search of ours. They had
never been shown to the statistical layer at all -- the positive-control artefact under
``runs/positive_control_net/`` carries a static verdict and nothing else -- so the saturation
question had never actually been put.

**The trial count is the whole argument, and it was fixed before this ran.** PI ruling of
2026-08-08: the primary figure deflates at ``N = 11``, the family's own count. That is consistent
with the standing design decision recorded in ``run_positive_control.py``'s docstring -- these
strategies "were never drawn from the generator's search, and are therefore not deflated by the
corpus trial count". The corpus count is reported beside it as a sensitivity because the
*difference between the two columns* is the measurement, and suppressing the second would make the
layer look more discriminating than the evidence supports.

**Units.** ``src/audit/stat.py`` works in per-observation Sharpe ratios. Annualised figures are
converted as ``SR_daily = SR_annual / sqrt(252)`` and ``V_daily = V_annual / 252``, and kurtosis is
non-excess, exactly as ``benchmarks/alphaaudit/statistical_deflation.md`` spells out. Passing an
annualised Sharpe alongside a daily observation count inflates the answer badly.

**Nothing frozen is modified.** This script imports ``src/audit/`` and runs it. The auditor has
been read-only since P2's release and stays that way.

**Development window only.** The holdout is closed and is not opened here.

Usage:
    python scripts/deflate_standard_factors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats as sps  # noqa: E402

# The control set and its loader are reused verbatim rather than restated, so this script cannot
# drift from the positive control it is extending.
from scripts.run_positive_control import FACTORS, load_strategy  # noqa: E402

from src.audit.stat import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import summarise  # noqa: E402

_log = get_logger(__name__)

OUT = Path("data/processed/standard_factor_deflation.json")

#: PI ruling of 2026-08-08. The primary trial count is the family's own size.
N_TRIALS_PRIMARY = len(FACTORS)

#: The AI corpus's honest ledger count, reported as a sensitivity and never as the headline.
N_TRIALS_CORPUS = 1887

#: The bar the AI corpus was held to, so the comparison is like for like.
DSR_BAR = 0.95

SESSIONS_PER_YEAR = 252


def _per_observation(returns: list[float]) -> dict[str, float]:
    """Sharpe, skew and non-excess kurtosis at daily frequency, computed not assumed."""
    array = np.asarray(returns, dtype=float)
    sd = float(array.std(ddof=1))
    return {
        "sharpe_per_observation": float(array.mean() / sd) if sd > 0 else 0.0,
        "skew": float(sps.skew(array, bias=False)),
        "kurtosis": float(sps.kurtosis(array, fisher=False, bias=False)),
        "n_observations": int(array.size),
    }


def main() -> int:
    configure_logging()
    cfg = load_config()
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    engine = BacktestEngine(panel=panel, calendar=calendar, universe=universe,
                            cost_model=CostModel(cfg.costs))

    rows: list[dict[str, Any]] = []
    series: list[np.ndarray] = []
    for name, family in FACTORS:
        result = engine.run(load_strategy(name)(), start=cfg.dates.dev_start,
                            end=cfg.dates.dev_end)
        moments = _per_observation(list(result.returns))
        rows.append({"name": name, "family": family,
                     "sharpe_annualised": summarise(result.returns)["sharpe_ratio"], **moments})
        series.append(np.asarray(result.returns, dtype=float))
        _log.info("%-32s SR/obs %+.6f  skew %+.3f  kurt %.2f",
                  name, moments["sharpe_per_observation"], moments["skew"],
                  moments["kurtosis"])

    # V is measured from this family's own dispersion, exactly as the corpus's V was measured from
    # the corpus. Assuming a value here would make the headline a statement about the assumption.
    trial_sharpes = np.array([r["sharpe_per_observation"] for r in rows])
    variance = float(trial_sharpes.var(ddof=1))

    for row in rows:
        row["psr_undeflated"] = probabilistic_sharpe_ratio(
            observed_sharpe=row["sharpe_per_observation"], benchmark_sharpe=0.0,
            n_observations=row["n_observations"], skew=row["skew"], kurtosis=row["kurtosis"])
        for label, n_trials in (("dsr_at_family_n", N_TRIALS_PRIMARY),
                                ("dsr_at_corpus_n", N_TRIALS_CORPUS)):
            row[label] = deflated_sharpe_ratio(
                observed_sharpe=row["sharpe_per_observation"], n_trials=n_trials,
                n_observations=row["n_observations"], skew=row["skew"],
                kurtosis=row["kurtosis"], variance_of_trial_sharpes=variance)

    # PBO over the family, on the same code path the twelve clean fixtures used.
    matrix = np.column_stack([s[:min(len(x) for x in series)] for s in series])
    pbo = probability_of_backtest_overfitting(matrix)

    payload = {
        "window": [str(cfg.dates.dev_start), str(cfg.dates.dev_end)],
        "note": ("development window only; the holdout is closed and was not opened. "
                 "Exploratory: this analysis appears in no pre-registration."),
        "n_factors": len(rows),
        "trial_sharpe_variance_per_observation": variance,
        "trial_sharpe_variance_annualised": variance * SESSIONS_PER_YEAR,
        "trial_sharpe_sd_annualised": float(np.sqrt(variance * SESSIONS_PER_YEAR)),
        "dsr_bar": DSR_BAR,
        "n_trials_primary": N_TRIALS_PRIMARY,
        "n_trials_corpus_sensitivity": N_TRIALS_CORPUS,
        "n_clearing_bar_at_family_n": sum(r["dsr_at_family_n"] >= DSR_BAR for r in rows),
        "n_clearing_bar_at_corpus_n": sum(r["dsr_at_corpus_n"] >= DSR_BAR for r in rows),
        "max_dsr_at_family_n": max(r["dsr_at_family_n"] for r in rows),
        "max_dsr_at_corpus_n": max(r["dsr_at_corpus_n"] for r in rows),
        "pbo": pbo,
        "pbo_matrix_shape": list(matrix.shape),
        "results": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _log.info("V annualised %.4f (sd %.4f)", payload["trial_sharpe_variance_annualised"],
              payload["trial_sharpe_sd_annualised"])
    _log.info("clearing %.2f bar: %d/%d at N=%d, %d/%d at N=%d",
              DSR_BAR, payload["n_clearing_bar_at_family_n"], len(rows), N_TRIALS_PRIMARY,
              payload["n_clearing_bar_at_corpus_n"], len(rows), N_TRIALS_CORPUS)
    _log.info("PBO %.6f over %s", pbo, payload["pbo_matrix_shape"])
    _log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
