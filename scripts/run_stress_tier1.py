"""Tier 1 stress run: every strategy re-decided under counterfactual histories.

This is the faithful tier. Each synthetic path is a full price panel rebuilt by
:mod:`src.stress.panel` along a regime-conditional bootstrap sequence, and every strategy is run
through the unmodified Phase 1.2 engine against it — so a strategy *re-forms its signals* under the
counterfactual history rather than having its realised returns reshuffled. That is what makes the
resulting fragility a property of the strategy and not of its return series.

**Population: the census, 185 strategies.** All 174 AlphaAudit survivors plus the 11 standard
academic factors. The PI's sample-size analysis put the minimum defensible sample at 50; the census
was chosen instead once the runtime was measured, which removes sampling error from every downstream
number and makes the Phase 2.2 predictor's training set 185 rows rather than 61.

**Parallel over paths, not over backtests.** A worker builds one synthetic panel and one engine,
then runs all 185 strategies against it. The alternative — one task per (strategy, path) — would
rebuild the panel and re-index the engine 18,500 times instead of 100.

``POLARS_MAX_THREADS=1`` is set for the workers deliberately. Measured on this machine, one polars
thread per worker is 17% *faster* per backtest than the default (mean 8.83s to 7.35s, slowest
32.33s to 16.75s) because 24 workers each spawning a thread pool contend with each other.

**The holdout is not reachable from here.** This script loads the development panel, which contains
no holdout row, and has no flag that would change that. The holdout is spent for this project.

Usage:
    python scripts/run_stress_tier1.py --paths 2 --workers 2 --strategies 3   # smoke test
    python scripts/run_stress_tier1.py --paths 100 --workers 24               # the full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from run_corpus_backtest import _instantiate, _load_strategy  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import Config, load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.costs.india import CostModel  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import summarise  # noqa: E402
from src.stress.crr import conditional_bootstrap_indices  # noqa: E402
from src.stress.panel import SyntheticPanelBuilder  # noqa: E402

_log = get_logger(__name__)

SURVIVORS = Path("benchmarks/alphaaudit/survivors")
#: The 11-factor positive control, named in scripts/run_positive_control.py and fixed before any
#: member was run. Referenced by name rather than by globbing tests/fixtures/clean, which holds 32
#: fixtures — 21 of which were never part of the control set.
FACTORS = (
    "momentum_skip_month", "dual_momentum_21_126", "low_volatility",
    "inverse_volatility_weighted", "mean_reversion_5d", "bollinger_reversion",
    "long_term_reversal_756d", "relative_strength_vs_universe", "equal_weight_universe",
    "high_volatility", "random_walk_baseline",
)

_STATE: dict[str, Any] = {}


# --- inputs ---------------------------------------------------------------------


def load_dev_panel(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The development price panel restricted to symbols that ever enter the universe.

    The restriction matches what ``BacktestEngine.__init__`` does anyway and is applied here so the
    dense matrices in :class:`SyntheticPanelBuilder` cover 185 symbols rather than 2,697.
    """
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    tradable = universe["symbol"].unique().to_list()
    panel = (
        pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
        .filter(
            (pl.col("session_date") >= cfg.dates.dev_start)
            & (pl.col("session_date") <= cfg.dates.dev_end)
            & pl.col("symbol").is_in(tradable)
        )
        .sort("session_date")
    )
    return panel, universe


def draw_paths(cfg: Config, sessions: list[date], n_paths: int) -> tuple[np.ndarray, float]:
    """Regime-conditional bootstrap index paths, one per synthetic history.

    Conditioning on the Phase 2.0 HMM labels is the PI's Fork 2 ruling, and the calibration shows
    why: conditional paths reproduce all nine reference moments of the real index series, while
    unconditional paths miss ``abs_autocorr_lag21``. The block length is read from the calibration
    artifact rather than restated here, so there is one place it can be wrong.
    """
    calibration = json.loads(
        (cfg.paths.data_processed / "crr_calibration.json").read_text(encoding="utf-8")
    )
    block_length = float(calibration["block_length"]["sessions"])

    labels = pl.read_parquet(cfg.paths.data_processed / "regime_labels.parquet")
    aligned = pl.DataFrame({"session_date": sessions[1:]}).join(
        labels.select("session_date", "state"), on="session_date", how="left"
    )
    if aligned["state"].null_count():
        raise ValueError(
            f"{aligned['state'].null_count()} of {aligned.height} return sessions have no regime "
            "label; conditioning on partial labels would silently drop them"
        )
    paths = conditional_bootstrap_indices(
        aligned["state"].to_numpy(), block_length, n_paths, seed=cfg.meta.seed
    )
    return paths, block_length


def strategy_paths(limit: int | None) -> list[tuple[str, str]]:
    """``(name, source)`` for the 185-strategy census, survivors first then the 11 factors."""
    survivors = sorted(p for p in SURVIVORS.glob("*.py") if p.stem != "__init__")
    entries = [(p.stem, str(p)) for p in survivors]
    entries += [(name, f"tests.fixtures.clean.{name}") for name in FACTORS]
    return entries[:limit] if limit else entries


# --- workers --------------------------------------------------------------------


def _initialise(out_dir: str) -> None:
    """Load the panel, calendar and strategies once per worker rather than once per path."""
    cfg = load_config()
    panel, universe = load_dev_panel(cfg)
    _STATE.update(
        cfg=cfg,
        universe=universe,
        calendar=load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet"),
        builder=SyntheticPanelBuilder(panel, universe),
        costs=CostModel(cfg.costs),
        out_dir=Path(out_dir),
    )


def _load_all(entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Import every strategy class once per worker. Failures are recorded, never raised."""
    import importlib

    loaded: dict[str, Any] = {}
    for name, source in entries:
        try:
            # Survivors are loose files loaded by path; the factor fixtures share helpers through
            # `from ._common import ...` and must be imported as package members, which a
            # file-path loader cannot resolve. Getting that wrong made all eleven fail identically
            # once before, and the failure looked like a pipeline fault.
            loaded[name] = (
                _load_strategy(Path(source)) if source.endswith(".py")
                else _class_from_module(importlib.import_module(source))
            )
        except Exception as exc:  # noqa: BLE001 - a strategy that will not import is a datum
            loaded[name] = f"{type(exc).__name__}: {exc}"[:200]
    return loaded


def _class_from_module(module: ModuleType) -> type[Strategy]:
    """The Strategy subclass a fixture module defines, matching run_positive_control's loader."""
    found = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
    ]
    if not found:
        raise TypeError("module defines no Strategy subclass")
    return found[-1]


def run_path(path_id: int, index_path: list[int], entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Build one synthetic panel and run every strategy against it. Never raises."""
    started = time.perf_counter()
    cfg: Config = _STATE["cfg"]
    out_dir: Path = _STATE["out_dir"]
    if "classes" not in _STATE:
        _STATE["classes"] = _load_all(entries)

    synthetic, diagnostics = _STATE["builder"].build(np.array(index_path, dtype=np.int64))
    engine = BacktestEngine(
        panel=synthetic, calendar=_STATE["calendar"], universe=_STATE["universe"],
        cost_model=_STATE["costs"],
    )

    results = [
        _run_one(engine, name, _STATE["classes"][name], cfg) for name, _ in entries
    ]
    payload = {
        "path_id": path_id,
        "seconds": time.perf_counter() - started,
        "diagnostics": diagnostics.as_dict(),
        "results": results,
    }
    _write_atomically(out_dir / f"path_{path_id:04d}.json", payload)
    return {"path_id": path_id, "seconds": payload["seconds"],
            "evaluated": sum(1 for r in results if r["outcome"] == "evaluated"),
            "diagnostics": payload["diagnostics"]}


def _run_one(
    engine: BacktestEngine, name: str, cls: type[Strategy] | str, cfg: Config
) -> dict[str, Any]:
    """One strategy against one synthetic panel. Untrusted code, so a failure is a result."""
    if isinstance(cls, str):
        return {"name": name, "outcome": "import_error", "error": cls}
    try:
        result = engine.run(_instantiate(cls), start=cfg.dates.dev_start, end=cfg.dates.dev_end)
    except Exception as exc:  # noqa: BLE001 - recorded rather than propagated, per Phase 1.4
        return {"name": name, "outcome": "runtime_error",
                "error": f"{type(exc).__name__}: {exc}"[:200]}

    net, gross = summarise(result.returns), summarise(result.gross_returns)
    n = len(result.returns) or 1
    return {
        "name": name,
        "outcome": "evaluated",
        "sharpe": net["sharpe_ratio"],
        "sharpe_gross": gross["sharpe_ratio"],
        "annualised_return": net["annualised_return"],
        "volatility": net["annualised_volatility"],
        "max_drawdown": net["max_drawdown"],
        "mean_turnover": float(sum(result.turnover) / n),
        "mean_cost": float(sum(result.costs) / n),
        "n_sessions": len(result.returns),
        "ruined_on": str(result.ruined_on) if result.ruined_on else None,
    }


def _write_atomically(target: Path, payload: dict[str, Any]) -> None:
    """Write via a temporary sibling and rename, so an interrupted run leaves no partial file.

    ``--resume`` decides what to skip by the presence of these files, so a half-written one would
    be silently treated as a completed path.
    """
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


# --- driver ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--strategies", type=int, default=None,
                        help="cap the strategy count; for the smoke test only")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="skip paths whose output file already exists")
    args = parser.parse_args()

    # One polars thread per worker: measured 17% faster than the default at this width.
    os.environ["POLARS_MAX_THREADS"] = "1"

    cfg = load_config()
    panel, _ = load_dev_panel(cfg)
    sessions = sorted(panel["session_date"].unique().to_list())
    paths, block_length = draw_paths(cfg, sessions, args.paths)
    entries = strategy_paths(args.strategies)

    out_dir = Path(args.out) if args.out else cfg.paths.runs / "tier1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        i for i in range(args.paths)
        if not (args.resume and (out_dir / f"path_{i:04d}.json").exists())
    ]
    _log.info(
        "%d strategies x %d paths (%d pending) on %d workers; block length %.2f, %d sessions",
        len(entries), args.paths, len(pending), args.workers, block_length, len(sessions),
    )

    with RunManifest(cfg, script="run_stress_tier1.py") as run:
        run.note("n_strategies", len(entries))
        run.note("n_paths", args.paths)
        run.note("block_length", block_length)
        run.note("workers", args.workers)
        completed = _dispatch(pending, paths, entries, args.workers, out_dir)
        run.note("paths_completed", len(completed))
    return 0 if len(completed) == len(pending) else 1


def _dispatch(
    pending: list[int], paths: np.ndarray, entries: list[tuple[str, str]],
    workers: int, out_dir: Path,
) -> list[dict[str, Any]]:
    """Run the pending paths, reporting each as it lands so progress is watchable."""
    progress = out_dir / "progress.jsonl"
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_initialise, initargs=(str(out_dir),)
    ) as pool:
        futures = {
            pool.submit(run_path, i, paths[i].tolist(), entries): i for i in pending
        }
        for future in as_completed(futures):
            try:
                summary = future.result()
            except Exception as exc:  # noqa: BLE001 - a lost path is reported, not fatal
                _log.error("path %d failed: %s: %s", futures[future], type(exc).__name__, exc)
                continue
            completed.append(summary)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True) + "\n")
            elapsed = time.perf_counter() - started
            _log.info(
                "path %d done in %.0fs (%d/%d, %.1f min elapsed, ~%.1f min left) "
                "| %d evaluated | missing members %.2f%%",
                summary["path_id"], summary["seconds"], len(completed), len(pending),
                elapsed / 60, elapsed / 60 * (len(pending) - len(completed)) / len(completed),
                summary["evaluated"], summary["diagnostics"]["missing_member_rate"] * 100,
            )
    return completed


if __name__ == "__main__":
    sys.exit(main())
