"""Run all three auditor layers over the corpus, recording each layer's verdict separately.

**The layers are never merged.** Each candidate gets a static verdict, a semantic verdict and a
statistical verdict, stored side by side. Collapsing them into "flagged by any layer" would destroy
the ablation the study is built to produce, and the Checkpoint 1.3 result shows why it would also be
wrong: `leak_full_sample_scaler` is caught by the static layer and correctly passed by the semantic
one, because its rationale honestly describes what its code does and only the fitting location is
at fault. The layers measure different things.

The semantic layer holds the GPU, so it runs serially after generation has finished. The static
layer is milliseconds per candidate. The statistical layer needs backtest returns and so consumes
the output of `run_corpus_backtest.py`.

Usage:
    python scripts/run_corpus_audit.py --corpus runs/<stamp>/candidates
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.audit.semantic import classify, is_available  # noqa: E402
from src.audit.stat import (  # noqa: E402
    TrialCounter,
    default_counter_path,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from src.audit.static import Severity, audit_file  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.eval.metrics import sharpe_ratio  # noqa: E402

_log = get_logger(__name__)


def extract_rationale(source: str) -> str:
    """The candidate's stated reasoning, for the semantic layer to read against its code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "rationale" for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            return value
    return ""


def static_layer(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Verdicts from `A_static`. Records every class found, not just the first."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        findings = audit_file(path)
        high = [f for f in findings if f.severity is Severity.HIGH]
        out[path.stem] = {
            "rejected": bool(high),
            "classes": sorted({f.leak_class.value for f in high}),
            "n_findings": len(findings),
            # Confidence for the coverage sweep: more independent high-severity classes is stronger
            # evidence than one rule firing repeatedly on the same construct.
            "confidence": float(len({f.leak_class.value for f in high})),
        }
    return out


def semantic_layer(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Verdicts from `A_sem`. Frozen prompt and model; nothing here was tuned for this corpus."""
    out: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        source = path.read_text(encoding="utf-8")
        rationale = extract_rationale(source)
        try:
            finding = classify(rationale, source)
        except Exception as exc:  # noqa: BLE001 - a model failure is a datum, not a crash
            out[path.stem] = {"rejected": False, "label": "error", "confidence": 0.0,
                              "error": f"{type(exc).__name__}: {exc}"[:200]}
            continue
        out[path.stem] = {
            "rejected": bool(finding.is_defect),
            "label": finding.label,
            "confidence": float(finding.confidence),
            "error": None,
        }
        if index % 25 == 0:
            rate = (time.perf_counter() - started) / index
            _log.info("semantic %d/%d, %.1fs each, ~%.0f min left",
                      index, len(paths), rate, rate * (len(paths) - index) / 60)
    return out


def statistical_layer(
    backtests: list[dict[str, Any]], n_trials: int
) -> tuple[dict[str, dict[str, Any]], float | None]:
    """Verdicts from `A_stat`: Deflated Sharpe per candidate, and PBO across the corpus."""
    evaluated = [b for b in backtests if b["outcome"] == "evaluated" and b["returns_path"]]
    if not evaluated:
        return {}, None

    series: dict[str, np.ndarray] = {}
    for record in evaluated:
        frame = pl.read_parquet(record["returns_path"])
        series[record["name"]] = frame["return"].to_numpy()

    # A common length is required to stack the matrix for PBO; the engine produces one row per
    # session, so any short series is a candidate that stopped early and is left out of PBO only.
    lengths = [len(v) for v in series.values()]
    common = max(set(lengths), key=lengths.count)
    stacked = np.column_stack([v for v in series.values() if len(v) == common])
    pbo = probability_of_backtest_overfitting(stacked) if stacked.shape[1] >= 4 else None

    sharpes = [sharpe_ratio(v.tolist()) for v in series.values()]
    variance_of_trials = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

    out: dict[str, dict[str, Any]] = {}
    for name, values in series.items():
        raw = sharpe_ratio(values.tolist())
        centred = values - values.mean()
        skew = float((centred**3).mean() / (values.std() ** 3)) if values.std() > 0 else 0.0
        kurt = float((centred**4).mean() / (values.std() ** 4)) if values.std() > 0 else 3.0
        dsr = deflated_sharpe_ratio(
            raw, n_trials=n_trials, n_observations=len(values),
            skew=skew, kurtosis=kurt, variance_of_trial_sharpes=variance_of_trials,
        )
        out[name] = {
            # DSR is the probability the true Sharpe exceeds the luck threshold. Below 0.95 the
            # candidate is not distinguishable from the best of N random tries.
            "rejected": bool(dsr < 0.95),
            "raw_sharpe": raw,
            "deflated_sharpe_probability": float(dsr),
            "confidence": float(1.0 - dsr),
        }
    return out, pbo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--skip-semantic", action="store_true",
                        help="static and statistical only; for a fast re-run")
    args = parser.parse_args()

    cfg = load_config()
    paths = sorted(args.corpus.glob("candidate_*.py"))
    if not paths:
        _log.error("no candidates under %s", args.corpus)
        return 1

    counter = TrialCounter(default_counter_path(cfg))
    n_trials = counter.verify()
    _log.info("trial ledger verified: %d trials", n_trials)

    _log.info("static layer over %d candidates", len(paths))
    static = static_layer(paths)

    semantic: dict[str, dict[str, Any]] = {}
    if not args.skip_semantic:
        if not is_available():
            _log.error("Ollama unreachable; semantic layer cannot run")
            return 1
        _log.info("semantic layer over %d candidates", len(paths))
        semantic = semantic_layer(paths)

    backtest_path = args.corpus.parent / "backtest_results.json"
    statistical: dict[str, dict[str, Any]] = {}
    pbo: float | None = None
    if backtest_path.exists():
        backtests = json.loads(backtest_path.read_text(encoding="utf-8"))
        statistical, pbo = statistical_layer(backtests, n_trials or len(paths))
    else:
        _log.warning("no backtest results at %s; statistical layer skipped", backtest_path)

    out = {
        "n_candidates": len(paths),
        "n_trials": n_trials,
        "pbo": pbo,
        "static": static,
        "semantic": semantic,
        "statistical": statistical,
    }
    (args.corpus.parent / "audit_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print(f"candidates: {len(paths)}   trials: {n_trials}")
    print(f"static rejected:      {sum(1 for v in static.values() if v['rejected'])}/{len(paths)}")
    if semantic:
        print(f"semantic rejected:    "
              f"{sum(1 for v in semantic.values() if v['rejected'])}/{len(semantic)}")
    if statistical:
        print(f"statistical rejected: "
              f"{sum(1 for v in statistical.values() if v['rejected'])}/{len(statistical)}")
    if pbo is not None:
        print(f"PBO across the corpus: {pbo:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
