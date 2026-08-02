"""Tier 2 stress run: 1,000 resampled-return paths per strategy, both bootstrap variants.

Reads the realised return series persisted by ``scripts/persist_real_returns.py`` and the Phase 2.0
regime labels, and writes one fragility record per (strategy, variant) to
``data/processed/tier2_fragility.json``.

Both variants are run in full because at Tier 2's cost there is no tradeoff between them, and
because Fork 2's ruling binds us to report the unconditional comparison alongside the conditional
result so a reader can see what conditioning changed (PI, Checkpoint 2.1).

Usage:
    python scripts/run_stress_tier2.py --paths 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.tier2 import FragilityResult, draw_paths, fragility  # noqa: E402

_log = get_logger(__name__)

SEED = 42
VARIANTS = ("conditional", "unconditional")


def _load() -> tuple[pl.DataFrame, dict[str, Any]]:
    """Return series joined to their regime labels, plus the calibrated block length."""
    cfg = load_config()
    returns = pl.read_parquet(cfg.paths.data_processed / "real_returns.parquet")
    labels = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet")
    calibration = json.loads(
        (cfg.paths.data_processed / "crr_calibration.json").read_text(encoding="utf-8")
    )
    joined = returns.join(labels.select("session_date", "state"), on="session_date", how="inner")
    dropped = returns.height - joined.height
    if dropped:
        # Not silently tolerated: a session with a return but no regime label means the label
        # coverage and the backtest window disagree, which would bias every regime slice.
        _log.warning("%d of %d return rows had no regime label and were dropped",
                     dropped, returns.height)
    return joined.sort("name", "session_date"), calibration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config()
    joined, calibration = _load()
    block_length = float(calibration["block_length"]["sessions"])
    names = joined["name"].unique(maintain_order=True).to_list()
    _log.info("tier 2: %d strategies, %d paths, block length %.2f sessions",
              len(names), args.paths, block_length)

    started = time.perf_counter()
    records: list[dict[str, object]] = []
    # Index paths depend only on the label sequence and its length, so strategies sharing a length
    # share their paths. Caching that both saves the draw and guarantees the sharing is exact.
    cache: dict[tuple[int, str], np.ndarray] = {}

    for name in names:
        block = joined.filter(pl.col("name") == name)
        returns = block["net_return"].to_numpy()
        labels = block["state"].to_numpy()
        knife = bool(block["knife_edge"][0])
        for variant in VARIANTS:
            key = (returns.shape[0], variant)
            if key not in cache:
                cache[key] = draw_paths(
                    labels, block_length, args.paths, SEED,
                    conditional=(variant == "conditional"),
                )
            result: FragilityResult = fragility(
                name, returns, labels, cache[key], variant=variant, knife_edge=knife
            )
            records.append(result.as_dict())
    wall = time.perf_counter() - started

    with RunManifest(cfg, script="run_stress_tier2.py") as run:
        payload = {
            "n_strategies": len(names),
            "n_paths": args.paths,
            "seed": SEED,
            "block_length_sessions": block_length,
            "variants": list(VARIANTS),
            "wall_clock_seconds": wall,
            "fragility": records,
        }
        out = cfg.paths.data_processed / "tier2_fragility.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        run.note("strategies", len(names))
        run.note("records", len(records))

    _summarise(records, wall, out)
    return 0


def _summarise(records: list[dict[str, Any]], wall: float, out: Path) -> None:
    """Print what a reader needs to judge the run without opening the artifact."""
    for variant in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        clean = [r for r in rows if not r["knife_edge"]]
        regimes = np.array([r["fragility_across_regimes"] for r in clean], dtype=float)
        paths = np.array([r["fragility_across_paths"] for r in clean], dtype=float)
        near_zero = sum(1 for r in clean if r["mean_is_near_zero"])
        print(f"\n  {variant}: {len(rows)} strategies ({len(clean)} excluding knife-edge)")
        print(f"    fragility across regimes   median {np.nanmedian(regimes):8.3f}"
              f"   n={np.isfinite(regimes).sum()}")
        print(f"    fragility across paths     median {np.nanmedian(paths):8.3f}"
              f"   n={np.isfinite(paths).sum()}")
        print(f"    mean near zero (F unstable) {near_zero} of {len(clean)}")
    print(f"\n  wall clock  {wall:.1f}s")
    print(f"  written to  {out}\n")


if __name__ == "__main__":
    sys.exit(main())
