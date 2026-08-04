"""Three measurements the frontier arms were missing, each using the benchmark's own definition.

Added 2026-08-04 after an audit found that the frontier arms had been put through each benchmark's
headline measurement but not its full set. All three are computed here rather than in three scripts
because none needs a re-run: every input is already on disk.

* **Fragility across regimes** --- P2's *primary* definition, the charter's
  ``Var_regime[pi] / |E_regime[pi]|`` on real regime labels. The arms had only the complementary
  across-paths measure, so "fragility" had been reported for them using the secondary statistic.
* **Flow-conditional capacity** --- P3's novel contribution: whether deployable capital differs
  between foreign inflow and outflow states. It was computed during the capacity run and never read.
* **Gross versus net** --- what Indian transaction costs do to the arms, including sign flips.

``across_regimes`` is imported from :mod:`src.stress.fragility` unchanged. Nothing here defines a
measurement of its own.

**Exploratory.** None of these appears in the generator-validation pre-registration.

Usage:
    python scripts/frontier_gap_measures.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.stress.fragility import across_regimes  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gpt", "claude", "gemini")
CRORE = 1e7


def _regime_fragility(arm: str, labels: pl.DataFrame) -> dict[str, Any]:
    """P2's primary fragility, per strategy, on the arm's realised returns."""
    run = ROOT / "runs" / f"frontier_{arm}"
    pooled = json.loads((run / "pooling.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for position, entry in enumerate(pooled["index"]):
        name = str(entry["candidate"]).removesuffix(".py")
        path = run / "backtests_development" / f"candidate_{position:03d}_returns.parquet"
        if not path.exists():
            continue
        # Joined on session_date rather than assumed aligned: a strategy ruined early has a
        # shorter series, and a positional join would silently label its returns with the wrong
        # regimes.
        joined = (
            pl.read_parquet(path)
            .select(["session_date", "return"])
            .join(labels, on="session_date", how="inner")
            .sort("session_date")
        )
        if joined.height < 2:
            continue
        result = across_regimes(
            name,
            joined["return"].to_numpy().astype(float),
            joined["state"].to_numpy(),
        )
        out[name] = {
            "fragility_across_regimes": result.fragility_across_regimes,
            "mean_regime_sharpe": result.mean_regime_sharpe,
            "regime_sharpe": result.regime_sharpe,
            "regime_sessions": result.regime_sessions,
            "mean_is_near_zero": result.mean_is_near_zero,
            "n_sessions": result.n_sessions,
        }
    return out


def _flow_capacity(arm: str) -> dict[str, Any]:
    """Per-strategy inflow/outflow capacity ratio, and the arm's distribution of it."""
    payload = json.loads(
        (ROOT / "runs" / f"frontier_{arm}" / "capacity.json").read_text(encoding="utf-8")
    )
    by_state: dict[str, dict[str, float]] = {}
    sessions: dict[str, dict[str, int]] = {}
    for row in payload["by_flow_state"]:
        by_state.setdefault(row["factor"], {})[row["flow_state"]] = row["median_capacity_inr"]
        sessions.setdefault(row["factor"], {})[row["flow_state"]] = row["n_sessions"]

    ratios: list[float] = []
    per_strategy: dict[str, Any] = {}
    for name, states in by_state.items():
        inflow, outflow = states.get("inflow"), states.get("outflow")
        if not inflow or not outflow or inflow <= 0:
            continue
        ratio = outflow / inflow
        ratios.append(ratio)
        per_strategy[name] = {
            "inflow_cr": inflow / CRORE,
            "outflow_cr": outflow / CRORE,
            "outflow_over_inflow": ratio,
            "sessions": sessions[name],
        }
    ordered = sorted(ratios)
    return {
        "n_strategies": len(ordered),
        "ratio_median": st.median(ordered) if ordered else None,
        "ratio_min": ordered[0] if ordered else None,
        "ratio_max": ordered[-1] if ordered else None,
        "per_strategy": per_strategy,
    }


def _gross_vs_net(arm: str) -> dict[str, Any]:
    """What Indian transaction costs did to the arm, including strategies whose sign flipped."""
    records = json.loads(
        (ROOT / "runs" / f"frontier_{arm}" / "backtest_results.json").read_text(encoding="utf-8")
    )
    usable = [
        r for r in records
        if r["outcome"] == "evaluated"
        and r.get("sharpe") is not None and r.get("sharpe_gross") is not None
    ]
    drops = [float(r["sharpe_gross"]) - float(r["sharpe"]) for r in usable]
    flipped = [
        r["name"] for r in usable
        if float(r["sharpe_gross"]) > 0 >= float(r["sharpe"])
    ]
    return {
        "n_evaluated": len(usable),
        "mean_sharpe_drop": float(np.mean(drops)) if drops else None,
        "median_sharpe_drop": float(np.median(drops)) if drops else None,
        "best_gross": max((float(r["sharpe_gross"]) for r in usable), default=None),
        "best_net": max((float(r["sharpe"]) for r in usable), default=None),
        "n_sign_positive_to_negative": len(flipped),
        "share_sign_positive_to_negative": len(flipped) / len(usable) if usable else None,
        "flipped_names": sorted(flipped),
        "median_turnover": float(np.median([float(r["mean_turnover"]) for r in usable]))
        if usable else None,
    }


def main() -> int:
    configure_logging()
    cfg = load_config()
    labels = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet").select(
        ["session_date", "state"]
    )
    local = json.loads(
        (cfg.paths.data_processed / "fragility.json").read_text(encoding="utf-8")
    )
    _log.info("local primary fragility: median %.3f across %d strategies",
              local["median_across_regimes"], local["n_primary"])

    results: dict[str, Any] = {
        "exploratory": True,
        "note": "Not pre-registered; added after a 2026-08-04 audit of measurement coverage.",
        "local_regime_fragility_median": local["median_across_regimes"],
        "local_near_zero_mean": local["n_flagged_near_zero_mean"],
        "local_n_primary": local["n_primary"],
        "arms": {},
    }

    for arm in ARMS:
        regime = _regime_fragility(arm, labels)
        values = sorted(
            r["fragility_across_regimes"] for r in regime.values()
            if r["fragility_across_regimes"] is not None
        )
        flow = _flow_capacity(arm)
        costs = _gross_vs_net(arm)
        results["arms"][arm] = {
            "regime_fragility": {
                "n": len(values),
                "median": st.median(values) if values else None,
                "min": values[0] if values else None,
                "max": values[-1] if values else None,
                "per_strategy": regime,
            },
            "flow_capacity": flow,
            "gross_vs_net": costs,
        }
        _log.info("%s:", arm)
        _log.info("  regime fragility  n=%2d  median %.3f  range %.3f-%.3f",
                  len(values), st.median(values), values[0], values[-1])
        _log.info("  flow capacity     n=%2d  outflow/inflow median %.3f  range %.3f-%.3f",
                  flow["n_strategies"], flow["ratio_median"],
                  flow["ratio_min"], flow["ratio_max"])
        _log.info("  costs             mean Sharpe drop %.4f  best gross %.4f -> net %.4f  "
                  "sign flips %d/%d",
                  costs["mean_sharpe_drop"], costs["best_gross"], costs["best_net"],
                  costs["n_sign_positive_to_negative"], costs["n_evaluated"])

    out = cfg.paths.data_processed / "frontier_gap_measures.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _log.info("wrote %s", out.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
