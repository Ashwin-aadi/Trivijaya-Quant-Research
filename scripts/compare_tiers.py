"""Answer the two questions that decide what Phase 2.2 is allowed to train on.

**1. Do the two fragility definitions agree?** The charter defines fragility across *regimes*;
Tier 1's stored output can only supply it across *paths*. If the two rank strategies the same, the
distinction is bookkeeping and Tier 1 needs no re-run. If they diverge, the choice is a real
methodological fork and belongs to the PI, with this measurement as the evidence.

**2. What does Tier 2's approximation cost?** Tier 1 lets a strategy re-decide under a
counterfactual price history; Tier 2 holds its decisions fixed and reshuffles their outcomes. Both
tiers cover the same strategies, so the gap between them is measurable rather than assumed — which
was the stated reason for running both (DECISIONS.md, Fork 1).

Rank correlation is Spearman's, computed here rather than imported: fragility is heavy-tailed and
a Pearson correlation would be dominated by a handful of strategies with near-zero mean returns.

Usage:
    python scripts/compare_tiers.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402

_log = get_logger(__name__)

NEAR_ZERO_MEAN = 0.05


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Rank correlation over the pairs where both values are finite, and how many those were."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    ranked = [np.argsort(np.argsort(v[ok])).astype(float) for v in (a, b)]
    centred = [r - r.mean() for r in ranked]
    denominator = np.sqrt((centred[0] ** 2).sum() * (centred[1] ** 2).sum())
    if denominator == 0.0:
        return float("nan"), int(ok.sum())
    return float((centred[0] * centred[1]).sum() / denominator), int(ok.sum())


def tier1_fragility() -> dict[str, dict[str, float]]:
    """Across-path fragility from the Tier 1 run: Var over the 100 paths, divided by |mean|.

    A strategy that failed on any path is kept, using the paths on which it succeeded, with its
    surviving count reported. Dropping it entirely would silently remove exactly the strategies
    whose behaviour is most history-dependent.
    """
    collected: dict[str, list[float]] = {}
    for file in sorted(glob.glob("runs/tier1/path_*.json")):
        payload = json.loads(Path(file).read_text(encoding="utf-8"))
        for record in payload["results"]:
            if record["outcome"] == "evaluated":
                collected.setdefault(record["name"], []).append(float(record["sharpe"]))

    out: dict[str, dict[str, float]] = {}
    for name, values in collected.items():
        array = np.array(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size < 2:
            continue
        mean = float(array.mean())
        out[name] = {
            "mean_sharpe": mean,
            "std_sharpe": float(array.std(ddof=1)),
            "fragility_across_paths": float(array.var(ddof=1)) / max(abs(mean), 1e-12),
            "n_paths": float(array.size),
            "mean_is_near_zero": float(abs(mean) < NEAR_ZERO_MEAN),
        }
    return out


def main() -> int:
    cfg = load_config()
    tier2 = json.loads(
        (cfg.paths.data_processed / "tier2_fragility.json").read_text(encoding="utf-8")
    )
    by_variant: dict[str, dict[str, dict[str, Any]]] = {
        variant: {r["name"]: r for r in tier2["fragility"] if r["variant"] == variant}
        for variant in tier2["variants"]
    }
    tier1 = tier1_fragility()
    report: dict[str, Any] = {"n_tier1": len(tier1), "n_tier2": len(by_variant["conditional"])}

    print("\n=== 1. DO THE TWO FRAGILITY DEFINITIONS AGREE? (Tier 2) ===")
    report["definition_agreement"] = {}
    for variant, records in by_variant.items():
        clean = [r for r in records.values() if not r["knife_edge"]]
        regimes = np.array([r["fragility_across_regimes"] for r in clean], dtype=float)
        paths = np.array([r["fragility_across_paths"] for r in clean], dtype=float)
        rho, n = _spearman(regimes, paths)
        report["definition_agreement"][variant] = {"spearman": rho, "n": n}
        print(f"  {variant:14s} Spearman rho = {rho:6.3f}   n = {n}")

    print("\n=== 2. WHAT DOES TIER 2'S APPROXIMATION COST? ===")
    report["tier_agreement"] = {}
    for variant, records in by_variant.items():
        shared = sorted(set(tier1) & {n for n, r in records.items() if not r["knife_edge"]})
        t1 = np.array([tier1[n]["fragility_across_paths"] for n in shared], dtype=float)
        t2 = np.array([records[n]["fragility_across_paths"] for n in shared], dtype=float)
        rho, n = _spearman(t1, t2)
        m1 = np.array([tier1[n]["mean_sharpe"] for n in shared], dtype=float)
        m2 = np.array([records[n]["mean_path_sharpe"] for n in shared], dtype=float)
        rho_mean, n_mean = _spearman(m1, m2)
        report["tier_agreement"][variant] = {
            "spearman_fragility": rho, "n": n,
            "spearman_mean_sharpe": rho_mean, "n_mean": n_mean,
            "median_tier1": float(np.nanmedian(t1)), "median_tier2": float(np.nanmedian(t2)),
        }
        print(f"  {variant:14s} fragility  rho = {rho:6.3f}  n = {n}"
              f"    median T1 {np.nanmedian(t1):7.3f}  T2 {np.nanmedian(t2):7.3f}")
        print(f"  {'':14s} mean Sharpe rho = {rho_mean:6.3f}  n = {n_mean}")

    print("\n=== 3. WHAT DID CONDITIONING CHANGE? ===")
    cond, uncond = by_variant["conditional"], by_variant["unconditional"]
    shared = sorted({n for n, r in cond.items() if not r["knife_edge"]})
    for field in ("fragility_across_regimes", "fragility_across_paths"):
        a = np.array([cond[n][field] for n in shared], dtype=float)
        b = np.array([uncond[n][field] for n in shared], dtype=float)
        rho, n = _spearman(a, b)
        ok = np.isfinite(a) & np.isfinite(b)
        print(f"  {field:28s} rho = {rho:6.3f}  n = {n}"
              f"   median cond {np.nanmedian(a):7.3f}  uncond {np.nanmedian(b):7.3f}")
        report.setdefault("conditioning_effect", {})[field] = {
            "spearman": rho, "n": n,
            "median_conditional": float(np.nanmedian(a)),
            "median_unconditional": float(np.nanmedian(b)),
            "mean_abs_difference": float(np.mean(np.abs(a[ok] - b[ok]))),
        }

    out = cfg.paths.data_processed / "tier_comparison.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n  written to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
