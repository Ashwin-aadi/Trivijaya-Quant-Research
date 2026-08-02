"""Measure, before the Tier 1 suite runs, the two things that would silently corrupt fragility.

Fragility is ``F(s) = Var_regime[pi(s)] / E_regime[pi(s)]`` — a variance across paths. Anything else
that varies across paths is therefore indistinguishable from the quantity being measured, and there
are exactly two candidates. This script quantifies both, on the full 185-strategy census, and times
the run while it is at it.

**A. Strategy nondeterminism.** A strategy that iterates a ``set`` or an unordered ``dict`` produces
different portfolios under different hash seeds, and Python randomises the seed per process. Every
worker in the Tier 1 pool is a separate process, so such a strategy would inject noise into the
path-to-path variance that has nothing to do with any regime. Group A and group B run the identical
real panel under two fixed, different hash seeds; any strategy whose Sharpe moves between them is
nondeterministic, and by how much.

**B. Reconstruction sensitivity.** Group C runs the *identity* synthetic panel — the reconstruction
of :mod:`src.stress.panel` along a no-op resampling, which agrees with the real panel to within
9e-15 relative on all 211,927 rows and every column. It shares group A's hash seed, so A versus C
isolates that residual floating-point difference. A strategy whose Sharpe moves materially between
them is amplifying 1e-15 into a visible number, which says the strategy sits on a knife edge and
not that the panel is wrong.

Both groups are read the same way: they bound how much of any measured fragility is an artefact.

Usage:
    python scripts/calibrate_tier1.py --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
from run_stress_tier1 import _load_all, _run_one, load_dev_panel, strategy_paths  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.stress.panel import SyntheticPanelBuilder  # noqa: E402

_log = get_logger(__name__)

#: Below this a Sharpe difference is float noise in the metric itself, not a behavioural change.
TOLERANCE = 1e-9

_STATE: dict[str, Any] = {}


def _initialise(identity: bool) -> None:
    """Build one engine per worker, over either the real panel or its identity reconstruction."""
    cfg = load_config()
    panel, universe = load_dev_panel(cfg)
    if identity:
        sessions = panel["session_date"].n_unique()
        panel, _ = SyntheticPanelBuilder(panel, universe).build(np.arange(sessions - 1))
    _STATE.update(
        cfg=cfg,
        engine=BacktestEngine(
            panel=panel,
            calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
            universe=universe,
            cost_model=CostModel(cfg.costs),
        ),
    )


def _one(name: str, source: str) -> dict[str, Any]:
    """Time and run a single strategy in a worker."""
    if "classes" not in _STATE:
        _STATE["classes"] = _load_all([(name, source)])
    elif name not in _STATE["classes"]:
        _STATE["classes"].update(_load_all([(name, source)]))
    started = time.perf_counter()
    outcome = _run_one(_STATE["engine"], name, _STATE["classes"][name], _STATE["cfg"])
    outcome["seconds"] = time.perf_counter() - started
    return outcome


def run_group(
    entries: list[tuple[str, str]], *, identity: bool, hash_seed: int, workers: int
) -> dict[str, dict[str, Any]]:
    """Run the whole census once, under a fixed hash seed, and return results by name.

    The seed is set in the environment before the pool is created, so every spawned worker inherits
    it. Fixing it is what makes the comparison between groups a controlled one — without it, each
    worker would draw its own seed and the two effects being separated here would be confounded.
    """
    os.environ["PYTHONHASHSEED"] = str(hash_seed)
    os.environ["POLARS_MAX_THREADS"] = "1"
    started = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialise, initargs=(identity,)
    ) as pool:
        futures = {pool.submit(_one, name, source): name for name, source in entries}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    _log.info(
        "group done: identity=%s seed=%d, %d strategies in %.1f min",
        identity, hash_seed, len(results), (time.perf_counter() - started) / 60,
    )
    return results


def compare(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> list[tuple[str, float, float, float]]:
    """Strategies whose Sharpe differs between two groups, largest difference first."""
    moved = []
    for name, a in left.items():
        b = right.get(name, {})
        if a.get("outcome") != "evaluated" or b.get("outcome") != "evaluated":
            continue
        if abs(a["sharpe"] - b["sharpe"]) > TOLERANCE:
            moved.append((name, a["sharpe"], b["sharpe"], abs(a["sharpe"] - b["sharpe"])))
    return sorted(moved, key=lambda row: -row[3])


def _report(label: str, moved: list[tuple[str, float, float, float]], total: int) -> None:
    print(f"\n{label}: {len(moved)} of {total} strategies moved")
    for name, a, b, delta in moved[:15]:
        print(f"    {name:18s} {a:10.4f} -> {b:10.4f}   |diff| {delta:.4f}")
    if len(moved) > 15:
        print(f"    ... and {len(moved) - 15} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--strategies", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    entries = strategy_paths(args.strategies, apply_exclusions=False)
    _log.info("calibrating on %d strategies, %d workers", len(entries), args.workers)

    with RunManifest(cfg, script="calibrate_tier1.py") as run:
        started = time.perf_counter()
        group_a = run_group(entries, identity=False, hash_seed=0, workers=args.workers)
        # The replicate: identical panel, identical hash seed, different processes. Anything that
        # moves here is nondeterministic for a reason that is not hash order — unseeded randomness
        # inside the strategy. Without this group the other two comparisons cannot be read, because
        # a strategy that is simply noisy shows up in both of them and is attributed to neither.
        group_a2 = run_group(entries, identity=False, hash_seed=0, workers=args.workers)
        group_b = run_group(entries, identity=False, hash_seed=1, workers=args.workers)
        group_c = run_group(entries, identity=True, hash_seed=0, workers=args.workers)
        wall = time.perf_counter() - started

        unseeded = compare(group_a, group_a2)
        nondeterministic = compare(group_a, group_b)
        reconstruction = compare(group_a, group_c)
        # A strategy that is noisy at a fixed seed cannot also be shown to be sensitive to the
        # reconstruction: its own noise explains the difference. Subtract it rather than
        # double-count it.
        noisy = {row[0] for row in unseeded} | {row[0] for row in nondeterministic}
        attributable = [row for row in reconstruction if row[0] not in noisy]
        payload = _summarise(entries, group_a, nondeterministic, reconstruction, wall, args.workers)
        payload["unseeded_at_fixed_seed"] = [
            {"name": n, "run_1": a, "run_2": b, "abs_diff": d} for n, a, b, d in unseeded
        ]
        payload["reconstruction_sensitive_excluding_noisy"] = [
            {"name": n, "real": a, "identity": b, "abs_diff": d} for n, a, b, d in attributable
        ]
        payload["deterministic_strategies"] = len(entries) - len(noisy)
        out = cfg.paths.data_processed / "tier1_calibration.json"
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        run.note("nondeterministic", len(nondeterministic))
        run.note("reconstruction_sensitive", len(reconstruction))

    _report("0. UNSEEDED RANDOMNESS (same panel, same seed, replicate)", unseeded, len(entries))
    _report("A. NONDETERMINISTIC (real panel, hash seed 0 vs 1)", nondeterministic, len(entries))
    _report(
        "C. RECONSTRUCTION-SENSITIVE, excluding strategies already shown noisy",
        attributable, len(entries),
    )
    print(f"\ndeterministic strategies: {payload['deterministic_strategies']} of {len(entries)}")
    _report(
        "B. RECONSTRUCTION-SENSITIVE (real vs identity, same seed)", reconstruction, len(entries)
    )
    print(f"\ntiming: {json.dumps(payload['timing'], indent=2)}")
    print(f"\nwritten to {cfg.paths.data_processed / 'tier1_calibration.json'}")
    return 0


def _summarise(
    entries: list[tuple[str, str]],
    group: dict[str, dict[str, Any]],
    nondeterministic: list[tuple[str, float, float, float]],
    reconstruction: list[tuple[str, float, float, float]],
    wall: float,
    workers: int,
) -> dict[str, Any]:
    """Assemble the artifact, including the projected Tier 1 runtime this measurement implies."""
    times = np.array([r["seconds"] for r in group.values() if "seconds" in r])
    failures = [(n, r.get("outcome"), r.get("error")) for n, r in group.items()
                if r.get("outcome") != "evaluated"]
    per_path = float(times.sum())
    return {
        "n_strategies": len(entries),
        "workers": workers,
        "wall_clock_seconds_for_three_groups": wall,
        "timing": {
            "mean_seconds_per_backtest": float(times.mean()),
            "median_seconds_per_backtest": float(np.median(times)),
            "max_seconds_per_backtest": float(times.max()),
            "serial_seconds_for_one_path": per_path,
            "projected_minutes_100_paths": per_path * 100 / workers / 60,
        },
        "failures": failures,
        "nondeterministic": [
            {"name": n, "seed_0": a, "seed_1": b, "abs_diff": d} for n, a, b, d in nondeterministic
        ],
        "reconstruction_sensitive": [
            {"name": n, "real": a, "identity": b, "abs_diff": d} for n, a, b, d in reconstruction
        ],
    }


if __name__ == "__main__":
    sys.exit(main())
