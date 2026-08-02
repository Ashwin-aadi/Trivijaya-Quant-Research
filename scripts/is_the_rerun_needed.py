"""Decide, by measurement rather than opinion, whether Tier 1 must be re-run for 5.4 hours.

The charter defines fragility across *regimes*. Tier 1 stored per-path summaries only, so recovering
that definition from Tier 1 appears to require re-running the whole suite while persisting daily
returns. Before spending the compute, two cheaper possibilities have to be ruled out, and both can
be tested against Tier 2's 1,000 paths, which are already on disk.

**Question A — do the counterfactual histories change the answer at all?** Across-regime fragility
can be computed directly on the *real* return series: slice each strategy's realised returns by
regime label and take the variance across regimes. That costs nothing; the data is already
persisted. If that plain historical figure ranks strategies the same as the bootstrap figure, the
resampling adds nothing to *this particular metric* and no re-run is warranted for it.

**Question B — how many paths does the estimate actually need?** If the across-regime ranking has
already converged by 25 paths, a re-run costs 1.4 hours rather than 5.4. Measured by subsampling
Tier 2's paths and comparing each subsample's ranking against the full 1,000-path answer.

Neither question can be answered from Tier 1, because Tier 1 cannot produce an across-regime number
at all. That is the point: Tier 2 is used here as an instrument for sizing a Tier 1 run.

Usage:
    python scripts/is_the_rerun_needed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.stress.tier2 import (  # noqa: E402
    MIN_REGIME_SESSIONS,
    NEAR_ZERO_MEAN,
    SESSIONS_PER_YEAR,
    draw_paths,
)

SUBSAMPLES = (10, 25, 50, 100, 250, 500)
SEED = 42


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    ranks = [np.argsort(np.argsort(v[ok])).astype(float) for v in (a, b)]
    centred = [r - r.mean() for r in ranks]
    denom = np.sqrt((centred[0] ** 2).sum() * (centred[1] ** 2).sum())
    rho = float((centred[0] * centred[1]).sum() / denom) if denom else float("nan")
    return rho, int(ok.sum())


def _sharpe(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    std = float(x.std(ddof=1))
    return float(x.mean() / std * np.sqrt(SESSIONS_PER_YEAR)) if std > 1e-10 else 0.0


def _across_regimes(values: np.ndarray) -> float:
    """Var over regime Sharpes divided by |mean|, matching src.stress.tier2."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    mean = float(finite.mean())
    return float(finite.var(ddof=1)) / max(abs(mean), NEAR_ZERO_MEAN if abs(mean) else 1e-12)


def real_series_fragility(returns: np.ndarray, labels: np.ndarray) -> float:
    """Across-regime fragility on the actual history — no resampling, no re-run, free."""
    per_regime = [
        _sharpe(returns[labels == label])
        for label in np.unique(labels)
        if int(np.sum(labels == label)) >= MIN_REGIME_SESSIONS
    ]
    return _across_regimes(np.array(per_regime, dtype=float))


def bootstrap_fragility(returns: np.ndarray, labels: np.ndarray, paths: np.ndarray) -> float:
    """Across-regime fragility averaged over resampled paths, as Tier 2 computes it."""
    sampled, sampled_labels = returns[paths], labels[paths]
    per_regime = []
    for label in np.unique(labels):
        if int(np.sum(labels == label)) < MIN_REGIME_SESSIONS:
            continue
        mask = sampled_labels == label
        values = [
            _sharpe(sampled[p][mask[p]])
            for p in range(sampled.shape[0])
            if int(mask[p].sum()) >= MIN_REGIME_SESSIONS
        ]
        if values:
            per_regime.append(float(np.mean(values)))
    return _across_regimes(np.array(per_regime, dtype=float))


def main() -> int:
    cfg = load_config()
    frame = pl.read_parquet(cfg.paths.data_processed / "real_returns.parquet")
    labels_frame = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet")
    joined = frame.join(
        labels_frame.select("session_date", "state"), on="session_date", how="inner"
    ).sort("name", "session_date")
    knife = {
        r["name"]
        for r in json.loads(
            Path("benchmarks/regimestress/knife_edge.json").read_text(encoding="utf-8")
        )["knife_edge"]
    }
    block = float(
        json.loads((cfg.paths.data_processed / "crr_calibration.json").read_text(encoding="utf-8"))
        ["block_length"]["sessions"]
    )
    names = [n for n in joined["name"].unique(maintain_order=True).to_list() if n not in knife]

    real, full = [], []
    subsampled: dict[int, list[float]] = {k: [] for k in SUBSAMPLES}
    cache: dict[int, np.ndarray] = {}
    for name in names:
        block_frame = joined.filter(pl.col("name") == name)
        returns = block_frame["net_return"].to_numpy()
        labels = block_frame["state"].to_numpy()
        if returns.shape[0] not in cache:
            cache[returns.shape[0]] = draw_paths(labels, block, 1000, SEED, conditional=True)
        paths = cache[returns.shape[0]]
        real.append(real_series_fragility(returns, labels))
        full.append(bootstrap_fragility(returns, labels, paths))
        for k in SUBSAMPLES:
            subsampled[k].append(bootstrap_fragility(returns, labels, paths[:k]))

    real_a, full_a = np.array(real), np.array(full)
    print(f"\n  strategies: {len(names)} (knife-edge excluded)\n")
    print("=== A. DO THE COUNTERFACTUAL HISTORIES CHANGE THE RANKING? ===")
    rho, n = _spearman(real_a, full_a)
    print(f"  real history vs 1,000 bootstrap paths   Spearman {rho:.3f}   n = {n}")
    print(f"  median fragility: real {np.nanmedian(real_a):.3f}   bootstrap "
          f"{np.nanmedian(full_a):.3f}")

    print("\n=== B. HOW MANY PATHS DOES THE RANKING NEED? ===")
    print(f"  {'paths':>6}  {'rho vs 1000':>12}  {'Tier 1 cost':>12}")
    for k in SUBSAMPLES:
        rho_k, _ = _spearman(np.array(subsampled[k]), full_a)
        print(f"  {k:>6}  {rho_k:>12.3f}  {k * 3.256 / 60:>10.1f} h")

    out = cfg.paths.data_processed / "rerun_decision.json"
    out.write_text(json.dumps({
        "n_strategies": len(names),
        "real_vs_bootstrap_spearman": _spearman(real_a, full_a)[0],
        "median_real": float(np.nanmedian(real_a)),
        "median_bootstrap": float(np.nanmedian(full_a)),
        "convergence": {
            str(k): _spearman(np.array(subsampled[k]), full_a)[0] for k in SUBSAMPLES
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n  written to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
