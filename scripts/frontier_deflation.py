"""Deflate one frontier arm at its own trial count, and compare it to matched-size M0 subsamples.

Implements the rule the PI settled in Amendment 2 of the generator-validation pre-registration:
per-arm deflation as the primary figure, plus a matched-N comparison against the local corpus, and
**both reported wherever either appears.** The amendment is explicit that no sentence may quote the
per-arm Deflated Sharpe without the matched-N result beside it, so this script emits them in one
payload and refuses to produce either alone.

Why the comparison exists. Deflating a 20-draw arm at N = 20 and the local corpus at N = 1,887
compares two search intensities, not two generators: the hurdle a strategy must clear is far lower
when the search was small. The matched comparison removes that by asking how often *the local model*
would clear the same bar if it had also been allowed only as many draws.

No measurement code is defined here. ``deflated_sharpe_ratio`` and ``sharpe_ratio`` are imported
from the frozen auditor, which has been read-only since P2's release.

Usage:
    python scripts/frontier_deflation.py --arm claude
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.audit.stat import deflated_sharpe_ratio  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.eval.metrics import sharpe_ratio  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]

#: The pre-registered bar. A strategy below it is not distinguishable from the best of N random
#: tries. Fixed by P1 and quoted here rather than re-chosen.
DSR_BAR = 0.95

#: Fixed in Amendment 2 and not re-drawn.
SUBSAMPLE_SEED = 42
N_SUBSAMPLES = 1000

#: The local corpus's rankable frame: strategies that executed *and* took a position. A strategy
#: that never traded has no return series to deflate, which is why rankable rather than executed is
#: the sampling frame.
POOLED_RESULTS = ROOT / "runs" / "pooled" / "backtest_results.json"


def _moments(values: np.ndarray) -> tuple[float, float, float]:
    """``(sharpe, skew, kurtosis)`` computed exactly as the corpus audit computes them."""
    raw = sharpe_ratio(values.tolist())
    spread = float(values.std())
    if spread <= 0.0:
        return raw, 0.0, 3.0
    centred = values - values.mean()
    return raw, float((centred**3).mean() / spread**3), float((centred**4).mean() / spread**4)


def _deflate(values: np.ndarray, n_trials: int, trial_variance: float) -> float:
    raw, skew, kurt = _moments(values)
    return float(
        deflated_sharpe_ratio(
            raw,
            n_trials=n_trials,
            n_observations=len(values),
            skew=skew,
            kurtosis=kurt,
            variance_of_trial_sharpes=trial_variance,
        )
    )


def arm_series(arm: str) -> dict[str, np.ndarray]:
    """Each strategy's realised net return series, keyed by the arm's own candidate name."""
    run = ROOT / "runs" / f"frontier_{arm}"
    pooled = json.loads((run / "pooling.json").read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    for position, entry in enumerate(pooled["index"]):
        path = run / "backtests_development" / f"candidate_{position:03d}_returns.parquet"
        if path.exists():
            out[str(entry["candidate"]).removesuffix(".py")] = (
                pl.read_parquet(path)["return"].to_numpy()
            )
    return out


def local_rankable() -> list[np.ndarray]:
    """The local corpus's 225 rankable series, the frame Amendment 2 samples from."""
    records = json.loads(POOLED_RESULTS.read_text(encoding="utf-8"))
    frame: list[np.ndarray] = []
    for record in records:
        if record["outcome"] != "evaluated" or not record.get("returns_path"):
            continue
        if not (record.get("mean_turnover") or 0) > 0:
            continue  # executed but never traded: nothing to deflate
        frame.append(pl.read_parquet(ROOT / record["returns_path"])["return"].to_numpy())
    return frame


def matched_comparison(frame: list[np.ndarray], size: int, n_trials: int) -> dict[str, Any]:
    """How often a random ``size``-draw from the local corpus clears the bar at ``n_trials``.

    The trial variance is recomputed within each subsample rather than taken from the whole corpus,
    because that is what the arm's own figure uses: a 20-draw arm knows only its own 20 Sharpes.
    Using the corpus-wide variance would hand the control information the arm does not have.
    """
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    cleared = 0
    best_per_draw: list[float] = []
    for _ in range(N_SUBSAMPLES):
        picks = [frame[i] for i in rng.choice(len(frame), size=size, replace=False)]
        sharpes = [sharpe_ratio(v.tolist()) for v in picks]
        variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        best = max(_deflate(v, n_trials, variance) for v in picks)
        best_per_draw.append(best)
        cleared += int(best >= DSR_BAR)
    return {
        "n_subsamples": N_SUBSAMPLES,
        "seed": SUBSAMPLE_SEED,
        "subsample_size": size,
        "deflated_at_n_trials": n_trials,
        "frame_size": len(frame),
        "subsamples_reaching_bar": cleared,
        "empirical_p": cleared / N_SUBSAMPLES,
        "median_best_dsr": float(np.median(best_per_draw)),
        "max_best_dsr": float(max(best_per_draw)),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--n-trials", type=int, action="append", default=None,
        help="trial count(s) to deflate at; defaults to the arm's size and to 5",
    )
    args = parser.parse_args()

    series = arm_series(args.arm)
    if not series:
        raise SystemExit(f"no return series found for arm {args.arm}")
    size = len(series)
    # Amendment 2 says "its own trial count" and illustrates it with N = 5, the arm size expected
    # when it was written. The arm as collected is four requests of five. Both readings are computed
    # and reported; which one is the pre-registered figure is a question for the PI, and it is left
    # open here rather than settled by this script.
    trial_counts = args.n_trials or sorted({size, 5})

    sharpes = [sharpe_ratio(v.tolist()) for v in series.values()]
    variance = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    frame = local_rankable()
    _log.info("arm %s: %d strategies; local rankable frame: %d", args.arm, size, len(frame))

    per_arm: dict[str, dict[str, Any]] = {}
    matched: dict[str, dict[str, Any]] = {}
    for n_trials in trial_counts:
        per_arm[str(n_trials)] = {
            name: {
                "raw_sharpe": _moments(values)[0],
                "deflated_sharpe_probability": _deflate(values, n_trials, variance),
            }
            for name, values in series.items()
        }
        cleared = [n for n, r in per_arm[str(n_trials)].items()
                   if r["deflated_sharpe_probability"] >= DSR_BAR]
        matched[str(n_trials)] = matched_comparison(frame, size, n_trials)
        _log.info(
            "  N=%d: %d of %d clear DSR>=%.2f; matched M0 draws clearing: %d/%d (p=%.3f)",
            n_trials, len(cleared), size, DSR_BAR,
            matched[str(n_trials)]["subsamples_reaching_bar"], N_SUBSAMPLES,
            matched[str(n_trials)]["empirical_p"],
        )
        if cleared:
            _log.warning("  clearing the bar at N=%d: %s", n_trials, ", ".join(sorted(cleared)))

    payload = {
        "arm": args.arm,
        "n_strategies": size,
        "dsr_bar": DSR_BAR,
        "variance_of_trial_sharpes": variance,
        "trial_counts_reported": trial_counts,
        "per_arm": per_arm,
        "matched_n": matched,
        "reporting_rule": (
            "Amendment 2: the per-arm figure may not be quoted without the matched-N result "
            "beside it."
        ),
    }
    out = ROOT / "runs" / f"frontier_{args.arm}" / "deflation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _log.info("  written to %s", out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
