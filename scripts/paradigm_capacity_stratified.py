"""Cross-arm capacity, split by whether the strategy actually deployed its capital.

Required by PREREGISTRATION.md Amendment 4 §4.3, which fixed this reporting rule before the
comparison was computed. Capacity is defined per rupee of AUM, not per rupee deployed, so a strategy
holding 1% of its account reports a capacity for an account that is not being invested. If arms
differ in how often they emit such strategies, a cross-arm capacity comparison measures how much
each arm invests rather than how scalable its output is.

They do differ. Measured near-cash shares run from 0.0% (G4) to 15.4% (G6), against the amendment's
predicted single-digit percentages -- a prediction this file's output shows to be wrong, and which
is left standing in the amendment as the record of it.

**Near-cash strategies are reported as a separate stratum, never dropped.** Dropping them after
seeing the capacity figures would be selection on the outcome, which is the pathology this lab
exists to detect.

Reads each arm's `capacity.json` and `exposure.json`; writes
`benchmarks/generationbench/capacity_stratified.json`.

Usage:
    python scripts/paradigm_capacity_stratified.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

CORPUS = Path("benchmarks/generationbench/corpus")
OUT = Path("benchmarks/generationbench/capacity_stratified.json")
ARM_ORDER = ("G1", "G2", "G4", "G5", "G6", "G7")


def _stratum(crores: list[float]) -> dict[str, Any]:
    """Median and range for one stratum, with its sample size beside every number."""
    if not crores:
        return {"n": 0, "median_crore": None, "min_crore": None, "max_crore": None}
    array = np.asarray(crores, dtype=float)
    return {
        "n": int(array.size),
        "median_crore": float(np.median(array)),
        "min_crore": float(array.min()),
        "max_crore": float(array.max()),
    }


def main() -> int:
    configure_logging()
    records = []
    for arm in ARM_ORDER:
        capacity = json.loads((CORPUS / arm / "capacity.json").read_text(encoding="utf-8"))
        exposure = json.loads((CORPUS / arm / "exposure.json").read_text(encoding="utf-8"))
        near_cash = {r["name"] for r in exposure["exposure"] if r["near_cash"]}

        deployed = [s["binding_capacity_inr"] / 1e7
                    for s in capacity["capacity"] if s["factor"] not in near_cash]
        cash = [s["binding_capacity_inr"] / 1e7
                for s in capacity["capacity"] if s["factor"] in near_cash]

        record = {
            "arm": arm,
            "paradigm": capacity["paradigm"],
            "near_cash_threshold": exposure["near_cash_threshold"],
            "near_cash_share": len(near_cash) / max(exposure["n_strategies"], 1),
            "deployed": _stratum(deployed),
            "near_cash": _stratum(cash),
            "all": _stratum(deployed + cash),
        }
        records.append(record)

        _log.info("%-3s deployed n=%3d median %6.2f cr (max %8.2f) | near-cash n=%2d "
                  "median %8.2f cr | unstratified median %6.2f cr",
                  arm, record["deployed"]["n"], record["deployed"]["median_crore"] or float("nan"),
                  record["deployed"]["max_crore"] or float("nan"), record["near_cash"]["n"],
                  record["near_cash"]["median_crore"] or float("nan"),
                  record["all"]["median_crore"] or float("nan"))

    OUT.write_text(json.dumps({
        "measure": "constraint-based deployment capacity, never impact erosion",
        "rule": "PREREGISTRATION.md Amendment 4 section 4.3, fixed before this was computed",
        "note": ("Near-cash strategies are reported as a separate stratum and never dropped. "
                 "Any arm difference that does not survive stratification is reported as not "
                 "surviving it."),
        "arms": records,
    }, indent=2, sort_keys=True), encoding="utf-8")
    _log.info("written to %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
