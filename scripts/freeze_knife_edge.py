"""Freeze the list of strategies whose Sharpe is not stable under 1e-15 input differences.

These strategies are deterministic — they gave the same answer twice at the same hash seed, and
the same answer across two different hash seeds, so they carry neither unseeded randomness nor
hash-order dependence. What they do instead is amplify. Run them over the *identity* reconstruction
of :mod:`src.stress.panel` — a synthetic panel resampled into the order the real panel was already
in, agreeing with it to 9e-15 relative on all 211,927 rows — and their Sharpe moves anyway.

Why they must come out of fragility. ``F(s) = Var_regime[pi(s)] / E_regime[pi(s)]`` reads
path-to-path variance as evidence about regimes. For a strategy sitting on a knife edge, some of
that variance is instead the floating-point residual of the reconstruction being amplified by a
discontinuous decision rule — a comparison that flips, a rank that swaps, a threshold crossed in
the fifteenth decimal. That contribution is real arithmetic, not a bug in the panel, but it is not
regime sensitivity and cannot be separated from it after the fact.

They are frozen rather than dropped. The PI's ruling is that they are excluded from fragility and
from the Phase 2.2 predictor's training set, and **reported separately** — a strategy whose
performance is not a stable function of its inputs is a finding about LLM-generated strategies,
not an inconvenience to be discarded quietly.

Reads ``data/processed/tier1_calibration.json`` (regenerate with ``scripts/calibrate_tier1.py``)
and writes ``benchmarks/regimestress/knife_edge.json``.

Usage:
    python scripts/freeze_knife_edge.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402

_log = get_logger(__name__)

OUT = Path("benchmarks/regimestress/knife_edge.json")
EXCLUDED = Path("benchmarks/regimestress/excluded_nondeterministic.json")

#: The eleven standard academic factors. Named here so the artifact can report whether the
#: exclusion costs us a reference baseline, which matters far more than losing one AI candidate:
#: the factors are the yardstick the AI strategies are measured against.
FACTORS = frozenset({
    "momentum_skip_month", "dual_momentum_21_126", "low_volatility",
    "inverse_volatility_weighted", "mean_reversion_5d", "bollinger_reversion",
    "long_term_reversal_756d", "relative_strength_vs_universe", "equal_weight_universe",
    "high_volatility", "random_walk_baseline",
})


def build() -> dict[str, object]:
    """Assemble the artifact from the calibration measurement."""
    cfg = load_config()
    source = cfg.paths.data_processed / "tier1_calibration.json"
    if not source.exists():
        raise FileNotFoundError(f"{source} is missing; run scripts/calibrate_tier1.py first")
    calibration = json.loads(source.read_text(encoding="utf-8"))

    # Already-noisy strategies were subtracted upstream: a strategy that moves at a fixed seed
    # cannot also be shown to be reconstruction-sensitive, because its own noise explains the
    # difference. What remains is attributable to the 9e-15 residual and nothing else.
    rows = sorted(
        calibration["reconstruction_sensitive_excluding_noisy"],
        key=lambda row: -float(row["abs_diff"]),
    )
    excluded = json.loads(EXCLUDED.read_text(encoding="utf-8"))["excluded"]
    nondeterministic = {record["name"] for record in excluded}
    factors_hit = sorted(r["name"] for r in rows if r["name"] in FACTORS)

    return {
        "purpose": (
            "Strategies excluded from RegimeStress fragility and from the Phase 2.2 predictor "
            "training set because their Sharpe is not stable under a 9e-15 relative change in "
            "their inputs. Reported separately per PI ruling of 2026-08-02, not discarded."
        ),
        "criterion": (
            "Deterministic by the excluded_nondeterministic.json test (stable across a replicate "
            "at fixed PYTHONHASHSEED and across two hash seeds), yet net Sharpe differs by more "
            "than 1e-9 between the real panel and its identity reconstruction."
        ),
        "measured_by": "scripts/calibrate_tier1.py, groups A and C",
        "measured_on": "2026-08-02",
        "frozen_on": datetime.now(UTC).date().isoformat(),
        "n_knife_edge": len(rows),
        "n_standard_factors_affected": len(factors_hit),
        "standard_factors_affected": factors_hit,
        "n_nondeterministic_separately_excluded": len(nondeterministic),
        "largest_swing": float(rows[0]["abs_diff"]) if rows else 0.0,
        "smallest_swing": float(rows[-1]["abs_diff"]) if rows else 0.0,
        "knife_edge": [
            {
                "name": r["name"],
                "sharpe_real_panel": float(r["real"]),
                "sharpe_identity_panel": float(r["identity"]),
                "abs_sharpe_swing": float(r["abs_diff"]),
                "is_standard_factor": r["name"] in FACTORS,
            }
            for r in rows
        ],
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _log.info(
        "froze %d knife-edge strategies (%d standard factors) to %s",
        payload["n_knife_edge"], payload["n_standard_factors_affected"], OUT,
    )
    print(f"\n  knife-edge strategies      {payload['n_knife_edge']}")
    print(f"  standard factors affected  {payload['n_standard_factors_affected']}"
          f"  {payload['standard_factors_affected']}")
    print(f"  largest Sharpe swing       {payload['largest_swing']:.4f}")
    print(f"  smallest Sharpe swing      {payload['smallest_swing']:.2e}")
    print(f"  written to                 {OUT}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
