"""Fragility targets for Phase 2.2: how much a strategy's performance depends on the environment.

Two quantities, deliberately kept distinct, because they answer different questions and the project
charter and the completed Tier 1 run disagree about which one "fragility" means.

``across_regimes`` — **primary, the charter's definition.**
    ``F = Var_r[pi_r] / |E_r[pi_r]|`` over the Phase 2.0 regime labels, computed on the strategy's
    *actual realised* return series. Does the strategy perform consistently across kinds of market?
    No resampling is involved, so it costs nothing beyond reading a parquet file.

``across_paths`` — **complementary, from Tier 1.**
    ``F = Var_p[pi_p] / |E_p[pi_p]|`` over the 100 synthetic price panels, on each of which the
    strategy genuinely re-decided. How much does the result depend on which counterfactual history
    the strategy met? Only Tier 1 can supply this; the cheap tier holds decisions fixed.

Using the real series for the primary target is not an assumption of convenience. It was tested
first: over 125 strategies the real-series figure ranks identically to the same statistic averaged
over 1,000 bootstrap paths at **Spearman 0.963** (``scripts/is_the_rerun_needed.py``), and a
100-path re-run would itself agree with a 1,000-path answer at only 0.952. The expensive experiment
validated the inexpensive computation rather than being replaced by it.

Both are ratios with a mean in the denominator, so a strategy whose mean performance sits near zero
carries a large ``F`` for an arithmetic reason rather than a behavioural one. Those are flagged, not
smoothed — see :data:`src.stress.tier2.NEAR_ZERO_MEAN`.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.stress.tier2 import MIN_REGIME_SESSIONS, _ratio, _sharpe

#: Tier 1's per-path summaries, one JSON file per synthetic price panel.
TIER1_GLOB = "runs/tier1/path_*.json"


@dataclass(frozen=True)
class RegimeFragility:
    """The charter's fragility for one strategy, with the sample size behind every component."""

    name: str
    fragility_across_regimes: float
    mean_regime_sharpe: float
    #: Realised Sharpe within each regime label. Index is the label value.
    regime_sharpe: dict[int, float]
    #: Sessions contributing to each entry above — no number without its sample size.
    regime_sessions: dict[int, int]
    n_sessions: int
    mean_is_near_zero: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fragility_across_regimes": self.fragility_across_regimes,
            "mean_regime_sharpe": self.mean_regime_sharpe,
            "regime_sharpe": {str(k): v for k, v in self.regime_sharpe.items()},
            "regime_sessions": {str(k): v for k, v in self.regime_sessions.items()},
            "n_sessions": self.n_sessions,
            "mean_is_near_zero": self.mean_is_near_zero,
        }


def across_regimes(name: str, returns: np.ndarray, labels: np.ndarray) -> RegimeFragility:
    """Fragility across regimes on the realised series — the charter's definition, no resampling.

    A regime thinner than :data:`MIN_REGIME_SESSIONS` contributes nothing rather than contributing
    a Sharpe ratio estimated from a handful of sessions. Dropping it is visible in
    ``regime_sessions``; silently including it would not be.
    """
    if returns.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{name}: {returns.shape[0]} returns against {labels.shape[0]} labels; "
            "the join lost or duplicated sessions"
        )

    per_regime: dict[int, float] = {}
    sessions: dict[int, int] = {}
    for label in np.unique(labels):
        slice_returns = returns[labels == label]
        if slice_returns.shape[0] < MIN_REGIME_SESSIONS:
            continue
        per_regime[int(label)] = float(_sharpe(slice_returns))
        sessions[int(label)] = int(slice_returns.shape[0])

    values = np.array(list(per_regime.values()), dtype=float)
    ratio, near_zero = _ratio(values)
    return RegimeFragility(
        name=name,
        fragility_across_regimes=ratio,
        mean_regime_sharpe=float(values.mean()) if values.size else float("nan"),
        regime_sharpe=per_regime,
        regime_sessions=sessions,
        n_sessions=int(returns.shape[0]),
        mean_is_near_zero=near_zero,
    )


def across_paths(pattern: str = TIER1_GLOB) -> dict[str, dict[str, float]]:
    """Across-path fragility from the Tier 1 run: variance over synthetic panels, over ``|mean|``.

    A strategy that failed on some paths is kept, using the paths on which it was evaluated, with
    the surviving count reported as ``n_paths``. Dropping such strategies entirely would remove
    exactly those whose behaviour is most history-dependent — the ones fragility exists to find.
    """
    collected: dict[str, list[float]] = {}
    for file in sorted(glob.glob(pattern)):
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
        ratio, near_zero = _ratio(array)
        out[name] = {
            "mean_path_sharpe": float(array.mean()),
            "std_path_sharpe": float(array.std(ddof=1)),
            "fragility_across_paths": ratio,
            "n_paths": float(array.size),
            "mean_is_near_zero": float(near_zero),
        }
    return out
