"""Assemble the Phase 2.2 fragility targets from work already on disk. Costs seconds, not hours.

Two targets per strategy, per the PI's Checkpoint 2.1 ruling:

* ``fragility_across_regimes`` — primary, the charter's definition, computed on the realised return
  series persisted by ``scripts/persist_real_returns.py``. No resampling; nothing to re-run.
* ``fragility_across_paths`` — complementary, read from the completed Tier 1 run.

Knife-edge strategies are excluded from the primary table and written to a separate file, so the
stability analysis reports them without their reconstruction sensitivity contaminating the ML
training set.

Usage:
    python scripts/build_fragility.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.fragility import across_paths, across_regimes  # noqa: E402

_log = logging.getLogger(__name__)

KNIFE_EDGE = Path("benchmarks/regimestress/knife_edge.json")


def _knife_edge_names() -> set[str]:
    payload = json.loads(KNIFE_EDGE.read_text(encoding="utf-8"))
    return {record["name"] for record in payload["knife_edge"]}


def main() -> int:
    configure_logging()
    cfg = load_config()
    processed = cfg.paths.data_processed

    returns = pl.read_parquet(processed / "real_returns.parquet")
    labels = pl.read_parquet(processed / "regime_labels.parquet").select("session_date", "state")
    joined = returns.join(labels, on="session_date", how="inner").sort("name", "session_date")
    if joined.height < returns.height:
        _log.warning(
            "%d of %d strategy-sessions carry no regime label and are excluded",
            returns.height - joined.height, returns.height,
        )

    knife = _knife_edge_names()
    tier1 = across_paths()
    _log.info("tier 1 supplied across-path fragility for %d strategies", len(tier1))

    with RunManifest(cfg, script="build_fragility.py") as run:
        rows: list[dict[str, object]] = []
        for name in joined["name"].unique(maintain_order=True).to_list():
            block = joined.filter(pl.col("name") == name)
            regime = across_regimes(
                name, block["net_return"].to_numpy(), block["state"].to_numpy()
            )
            record: dict[str, object] = dict(regime.as_dict())
            record["knife_edge"] = name in knife
            paths = tier1.get(name)
            record["fragility_across_paths"] = (
                paths["fragility_across_paths"] if paths else float("nan")
            )
            record["mean_path_sharpe"] = paths["mean_path_sharpe"] if paths else float("nan")
            record["n_paths"] = int(paths["n_paths"]) if paths else 0
            rows.append(record)

        primary = [r for r in rows if not r["knife_edge"]]
        excluded = [r for r in rows if r["knife_edge"]]
        both = [r for r in primary if r["n_paths"]]
        values = np.array([r["fragility_across_regimes"] for r in primary], dtype=float)
        finite = values[np.isfinite(values)]

        payload = {
            "n_strategies": len(rows),
            "n_primary": len(primary),
            "n_knife_edge_excluded": len(excluded),
            "n_with_both_definitions": len(both),
            "n_flagged_near_zero_mean": sum(bool(r["mean_is_near_zero"]) for r in primary),
            "median_across_regimes": float(np.median(finite)) if finite.size else float("nan"),
            "iqr_across_regimes": [
                float(np.percentile(finite, 25)), float(np.percentile(finite, 75)),
            ] if finite.size else [float("nan"), float("nan")],
            "primary": primary,
            "knife_edge_excluded": excluded,
        }
        out = processed / "fragility.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        run.note("output", str(out))
        run.note("n_primary", len(primary))

    _log.info(
        "%d strategies: %d primary, %d knife-edge excluded, %d with both definitions",
        len(rows), len(primary), len(excluded), len(both),
    )
    _log.info(
        "across-regime fragility  median %.4f  IQR %.4f-%.4f  (n=%d finite of %d)",
        payload["median_across_regimes"], payload["iqr_across_regimes"][0],
        payload["iqr_across_regimes"][1], finite.size, len(primary),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
