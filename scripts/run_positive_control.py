"""Run standard factor strategies through the same pipeline as the AI corpus, as a positive control.

**The question this answers.** The AI corpus was weak: 40% executed, 65% of those never traded, and
the tradeable remainder collapsed out of sample. Two explanations fit that equally well — the
generator is weak, or the benchmark is broken. They are distinguished by running strategies whose
behaviour is known in advance. If textbook factors execute, trade, and produce sensible numbers on
the same code path, the pipeline measures correctly and the weakness belongs to the generator.

**Development period only.** The holdout was evaluated once, under authorisation, and is spent for
this project. Nothing here may touch it, and nothing here does: the engine loads the development
panel, which contains no holdout row.

**Not part of the frozen benchmark.** These are a reference point against which the AI corpus's
weakness is measured. They were hand-written as Phase 1.2 test fixtures, they were never drawn from
the generator's search, and they are therefore not deflated by the corpus trial count and not
entered into any leaderboard.

**What is not covered.** Value and quality in their standard forms need fundamental data — book
value, earnings, accruals, return on equity — which this repository does not have. Only OHLCV is
available. `long_term_reversal_756d` is included as the documented *price-based* proxy for value
(De Bondt and Thaler 1985) and is labelled as a proxy, not as value. **Quality has no price-based
proxy and is simply absent.** Reporting a fabricated stand-in would defeat the purpose of a control.

Usage:
    python scripts/run_positive_control.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import polars as pl  # noqa: E402

from src.audit.semantic import classify, is_available  # noqa: E402
from src.audit.static import Severity, audit_file  # noqa: E402
from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.data.calendar import load_calendar  # noqa: E402
from src.eval.metrics import summarise  # noqa: E402

_log = get_logger(__name__)

FIXTURES = Path("tests/fixtures/clean")

#: The control set, each tagged with the factor family it represents. Chosen before any of them was
#: run through this script, so the set is not selected on its results.
FACTORS: tuple[tuple[str, str], ...] = (
    ("momentum_skip_month", "momentum (12-1, skipping the most recent month)"),
    ("dual_momentum_21_126", "momentum (dual horizon)"),
    ("low_volatility", "low volatility"),
    ("inverse_volatility_weighted", "low volatility (weighting form)"),
    ("mean_reversion_5d", "short-term reversal"),
    ("bollinger_reversion", "short-term reversal (band form)"),
    ("long_term_reversal_756d", "PROXY for value - price-based, not fundamental"),
    ("relative_strength_vs_universe", "cross-sectional relative strength"),
    ("equal_weight_universe", "market reference - should track the index"),
    ("high_volatility", "inverted control - should trail low volatility"),
    ("random_walk_baseline", "null control - should be indistinguishable from noise"),
)


def load_strategy(name: str) -> type[Strategy]:
    """Import a fixture by its package path and return the Strategy subclass it defines.

    Imported as a package member, not as a loose file. These fixtures share helpers through
    ``from ._common import ...``, which a file-path loader cannot resolve — it has no parent package
    to resolve the leading dot against. Loading them that way made all eleven raise identically, and
    the failure looked exactly like a pipeline fault when it was a defect in this loader.
    """
    module = importlib.import_module(f"tests.fixtures.clean.{name}")
    subclasses = [
        obj for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy
        and obj.__module__ == module.__name__
    ]
    if not subclasses:
        raise TypeError(f"{name} defines no Strategy subclass")
    return subclasses[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/positive_control"))
    args = parser.parse_args()

    cfg = load_config()
    panel = pl.read_parquet(cfg.paths.data_processed / "prices_adjusted.parquet")
    universe = pl.read_parquet(cfg.paths.data_processed / "universe.parquet")
    calendar = load_calendar(cfg.paths.data_raw / "calendar_cnx100.parquet")
    engine = BacktestEngine(panel=panel, calendar=calendar, universe=universe)

    semantic_ready = not args.skip_semantic and is_available()
    if not args.skip_semantic and not semantic_ready:
        _log.warning("Ollama unreachable; semantic verdicts will be omitted rather than faked")

    rows: list[dict[str, Any]] = []
    for name, family in FACTORS:
        path = FIXTURES / f"{name}.py"
        if not path.exists():
            _log.error("%s missing", path)
            return 1

        source = path.read_text(encoding="utf-8")
        findings = audit_file(path)
        high = [f for f in findings if f.severity is Severity.HIGH]

        semantic_label = None
        if semantic_ready:
            rationale = getattr(load_strategy(name), "rationale", "")
            try:
                semantic_label = classify(rationale, source).label
            except Exception as exc:  # noqa: BLE001 - a model failure is a datum
                semantic_label = f"error:{type(exc).__name__}"

        try:
            result = engine.run(
                load_strategy(name)(), start=cfg.dates.dev_start, end=cfg.dates.dev_end
            )
        except Exception as exc:  # noqa: BLE001 - recorded, exactly as for the AI corpus
            rows.append({"name": name, "family": family, "outcome": "runtime_error",
                         "error": f"{type(exc).__name__}: {exc}"[:200], "sharpe": None,
                         "static_rejected": bool(high), "semantic_label": semantic_label})
            _log.error("%s failed: %s", name, exc)
            continue

        stats = summarise(result.returns)
        traded = sum(1 for r in result.returns if abs(r) > 1e-12)
        rows.append({
            "name": name,
            "family": family,
            "outcome": "evaluated",
            "error": None,
            "sharpe": stats["sharpe_ratio"],
            "annualised_return": stats["annualised_return"],
            "volatility": stats["annualised_volatility"],
            "max_drawdown": stats["max_drawdown"],
            "n_sessions": len(result.returns),
            "n_active_sessions": traded,
            "flat": traded == 0,
            "static_rejected": bool(high),
            "static_classes": sorted({f.leak_class.value for f in high}),
            "semantic_label": semantic_label,
        })
        _log.info("%s: Sharpe %.4f over %d sessions (%d active)",
                  name, stats["sharpe_ratio"], len(result.returns), traded)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "positive_control.json").write_text(
        json.dumps({"window": [str(cfg.dates.dev_start), str(cfg.dates.dev_end)],
                    "note": "development period only; the holdout is spent and untouched here",
                    "results": rows}, indent=2),
        encoding="utf-8",
    )

    evaluated = [r for r in rows if r["outcome"] == "evaluated"]
    flat = [r for r in evaluated if r["flat"]]
    print(f"\n{'strategy':<32} {'Sharpe':>8} {'active':>7}  {'static':>7} {'semantic':<28} family")
    for r in sorted(rows, key=lambda x: -(x["sharpe"] or -99)):
        sharpe = f"{r['sharpe']:.4f}" if r["sharpe"] is not None else "FAILED"
        active = r.get("n_active_sessions", 0)
        verdict = "REJECT" if r["static_rejected"] else "pass"
        print(f"{r['name']:<32} {sharpe:>8} {active:>7}  {verdict:>7} "
              f"{str(r['semantic_label']):<28} {r['family']}")

    print(f"\nexecuted: {len(evaluated)}/{len(rows)}   flat: {len(flat)}")
    if len(evaluated) != len(rows) or flat:
        print("\nHALT CONDITION: a standard factor failed to execute or never traded. That "
              "implicates the pipeline, not the generator. Do not proceed to the write-up.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
