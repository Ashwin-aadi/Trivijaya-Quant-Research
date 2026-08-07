"""Tier 2 fragility for each P4 arm, at 1,000 resampled-return paths, both bootstrap variants.

**Why the cheap tier, stated plainly, because the honest sentence and the flattering one differ.**
P2 measured that the tiers agree on mean performance at Spearman 0.897 and on *fragility* at only
0.620, and concluded in as many words that "Tier 2 cannot be substituted for Tier 1 when fragility
is the quantity of interest". Tier 1 is nevertheless not available here: P2 spent 124.5 CPU-hours on
125 strategies, so P4's 315 would cost upwards of 250 on consumer hardware at a zero budget.

So this is a **resource limitation honestly recorded, not a claim that the tiers agree**. P4's
fragility figures are not interchangeable with P2's and must never be reported as though they were.
The PI ruled on this on 2026-08-07 with the cost in front of them.

`draw_paths` and `fragility` are imported from :mod:`src.stress.tier2` unchanged, and the block
length comes from P2's committed CRR calibration, so the measurement is P2's -- only the population
is new.

Knife-edge strategies are carried through and flagged rather than dropped, exactly as P2 does: a
strategy whose returns are not a stable function of its inputs would have that instability resampled
into every path, and it is reported separately rather than discarded quietly.

Reads each arm's `backtest_results.json` and `calibration.json`; writes
`benchmarks/generationbench/corpus/<arm>/fragility.json`.

Usage:
    python scripts/paradigm_stress.py --paths 1000
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
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.generate.paradigms.registry import ARMS  # noqa: E402
from src.stress.tier2 import FragilityResult, draw_paths, fragility  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
SEED = 42
VARIANTS = ("conditional", "unconditional")


def arm_returns(arm: str, labels: pl.DataFrame) -> pl.DataFrame:
    """Realised net returns for every traded candidate in `arm`, joined to its regime label.

    The backtests already persisted one parquet per candidate, so nothing is re-run here. A session
    carrying a return but no regime label means the label coverage and the backtest window disagree,
    which would bias every regime slice, so the shortfall is reported rather than absorbed.
    """
    results = json.loads((CORPUS / arm / "backtest_results.json").read_text(encoding="utf-8"))
    knife = _knife_edge_names(arm)

    frames: list[pl.DataFrame] = []
    for row in results:
        if row["outcome"] != "evaluated" or not (row.get("mean_turnover") or 0) > 0:
            continue
        if not row.get("returns_path") or not Path(row["returns_path"]).exists():
            continue
        frames.append(
            pl.read_parquet(row["returns_path"])
            .select("session_date", pl.col("return").alias("net_return"))
            .with_columns(name=pl.lit(row["name"]),
                          knife_edge=pl.lit(row["name"] in knife))
        )
    if not frames:
        return pl.DataFrame()

    stacked = pl.concat(frames)
    joined = stacked.join(labels.select("session_date", "state"), on="session_date", how="inner")
    if joined.height < stacked.height:
        _log.warning("%s: %d of %d return rows had no regime label and were dropped",
                     arm, stacked.height - joined.height, stacked.height)
    return joined.sort("name", "session_date")


def _knife_edge_names(arm: str) -> set[str]:
    """Names excluded from primary fragility, from this arm's calibration."""
    path = CORPUS / arm / "calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/paradigm_calibrate.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"] for row in payload.get("knife_edge", [])}


def stress_arm(arm: str, labels: pl.DataFrame, block_length: float, paths: int) -> dict[str, Any]:
    """Fragility for one arm under both bootstrap variants."""
    joined = arm_returns(arm, labels)
    if joined.is_empty():
        return {"arm": arm, "n_strategies": 0, "fragility": []}

    names = joined["name"].unique(maintain_order=True).to_list()
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    # Index paths depend only on the label sequence and its length, so strategies sharing a length
    # share their paths. Caching that saves the draw and makes the sharing exact rather than likely.
    cache: dict[tuple[int, str], np.ndarray] = {}

    for name in names:
        block = joined.filter(pl.col("name") == name)
        returns, states = block["net_return"].to_numpy(), block["state"].to_numpy()
        knife = bool(block["knife_edge"][0])
        for variant in VARIANTS:
            key = (returns.shape[0], variant)
            if key not in cache:
                cache[key] = draw_paths(states, block_length, paths, SEED,
                                        conditional=(variant == "conditional"))
            result: FragilityResult = fragility(name, returns, states, cache[key],
                                                variant=variant, knife_edge=knife)
            records.append(result.as_dict())

    clean = [r for r in records if r["variant"] == "conditional" and not r["knife_edge"]]
    across = [r["fragility_across_regimes"] for r in clean]
    median = float(np.nanmedian(across)) if across else float("nan")
    _log.info("%-3s %3d strategies (%d excluding knife-edge) in %.1fs; median across-regime F %.3f",
              arm, len(names), len(clean), time.perf_counter() - started, median)

    return {
        "arm": arm, "paradigm": ARMS[arm], "n_strategies": len(names),
        "n_paths": paths, "seed": SEED, "block_length_sessions": block_length,
        "variants": list(VARIANTS),
        "tier": 2,
        "tier_caveat": (
            "Tier 2 only. P2 measured tier agreement on fragility at Spearman 0.620 (n = 125) and "
            "concluded tier 2 cannot substitute for tier 1 when fragility is the quantity of "
            "interest. Tier 1 was not affordable for this corpus (P2 spent 124.5 CPU-hours on 125 "
            "strategies). These figures are not interchangeable with P2's published fragility."
        ),
        "fragility": records,
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    args = parser.parse_args()

    cfg = load_config()
    labels = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet")
    calibration = json.loads(
        (cfg.paths.data_processed / "crr_calibration.json").read_text(encoding="utf-8"))
    block_length = float(calibration["block_length"]["sessions"])
    _log.info("tier 2, %d paths, block length %.2f sessions (P2's committed calibration)",
              args.paths, block_length)

    for arm in ([args.arm] if args.arm else list(ARMS)):
        payload = stress_arm(arm, labels, block_length, args.paths)
        out = CORPUS / arm / "fragility.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _log.info("written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
