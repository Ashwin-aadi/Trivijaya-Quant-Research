"""Produce every number the statistical auditor contributes to the Checkpoint 1.3 report.

This is a measurement script. It runs the existing deflation code in ``src/audit/stat.py`` against
real backtests and prints the results; it fits nothing, tunes nothing, and takes no parameter that
was chosen after seeing an output.

Five sections, in the order the checkpoint reports them:

1. **The worked example.** A hypothetical strategy showing an annualised Sharpe of 2.0 over ten
   years, found after 200 trials. Printed with its intermediate quantities — the luck threshold and
   the Sharpe standard error — because the checkpoint has to explain the drop, not just quote it.
2. **A trial-count sweep.** The same strategy at trial counts from 1 to 1000, so the cost of
   searching harder is visible as a curve rather than a single number.
3. **A live positive control.** ``EqualWeightUniverse`` posts a high Sharpe over calendar 2023 and
   a mediocre one over the full development window. Both are measured here by running the engine,
   and the 2023 figure is then put through the deflation to show what an honestly-stated window and
   trial count do to it. Its skewness and kurtosis are computed from the realised return series,
   not assumed.
4. **Probability of Backtest Overfitting** over a corpus of clean fixtures run on the development
   window.
5. **A trial-counter demonstration.** Records trials including failures, verifies the chain, then
   hand-edits one line and shows verification failing. Runs against a temporary path; the project's
   real ledger is never opened.

Usage:
    python scripts/run_stat_audit.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

from src.audit.stat import (  # noqa: E402
    TamperError,
    TrialCounter,
    _sharpe_standard_error,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from src.backtest.engine import BacktestEngine  # noqa: E402
from src.backtest.strategy import Strategy  # noqa: E402
from src.common.config import Config, load_config  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.common.seeding import seed_everything  # noqa: E402
from src.data.calendar import TradingCalendar, load_calendar  # noqa: E402
from src.eval.metrics import sharpe_ratio  # noqa: E402

# Sessions per year used to move between annualised figures a human quotes and the per-observation
# units src/audit/stat.py works in. Matches the convention stated in that module's docstring and
# used throughout tests/audit/test_stat.py. Note this is 252, whereas src/eval/metrics.py
# annualises with 250; the difference is reported rather than reconciled, see _moments below.
SESSIONS_PER_YEAR = 252

CLEAN_DIR = Path("tests/fixtures/clean")

# Fixtures used for the PBO corpus: the first twelve in sorted filename order. Chosen by a rule
# fixed before any of them was run, so the corpus cannot have been selected on its answer.
PBO_CORPUS_SIZE = 12

# A fixture holding nothing on more than this share of sessions is reported separately in the PBO
# sensitivity. Half is the threshold because past it the series describes the data's start date more
# than the strategy. Fixed before the corpus was run; the headline PBO does not apply it.
MAX_ZERO_FRACTION = 0.5

# The window whose Sharpe the live control is deflated over. Calendar 2023 is one year of the
# development period; it is quoted because it is the sub-window on which this strategy looks best,
# which is exactly the situation deflation exists to price.
CONTROL_START = date(2023, 1, 1)
CONTROL_END = date(2023, 12, 31)


def _load_clean_fixtures() -> list[tuple[str, Strategy]]:
    """Instantiate every clean fixture, in sorted filename order.

    Discovery matches tests/audit/test_static.py: every ``*.py`` in the clean fixture directory
    that does not start with an underscore. The strategy class is the single Strategy subclass
    defined in that module.
    """
    fixtures: list[tuple[str, Strategy]] = []
    for path in sorted(CLEAN_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module(f"tests.fixtures.clean.{path.stem}")
        classes = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Strategy) and obj.__module__ == module.__name__
        ]
        if len(classes) != 1:
            raise RuntimeError(f"{path.name}: expected one Strategy subclass, found {len(classes)}")
        fixtures.append((path.stem, classes[0]()))
    return fixtures


def _moments(returns: list[float]) -> dict[str, float]:
    """Per-observation Sharpe, skewness and non-excess kurtosis of a realised return series.

    The Sharpe here is the plain ``mean / sample sd`` at daily frequency, which is what the DSR
    formula is defined on. It is *not* ``annualised / sqrt(252)``: src/eval/metrics.py annualises
    the numerator geometrically over 250 sessions, so the two differ slightly. Both are printed so
    the difference is visible rather than hidden.

    Kurtosis is requested non-excess (``fisher=False``), the convention
    :func:`src.audit.stat.probabilistic_sharpe_ratio` requires and actively rejects the opposite of.
    """
    array = np.asarray(returns, dtype=np.float64)
    standard_deviation = float(array.std(ddof=1))
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "sd": standard_deviation,
        "sharpe_per_observation": float(array.mean() / standard_deviation),
        "sharpe_annualised_metrics": sharpe_ratio(returns),
        "skew": float(stats.skew(array, bias=False)),
        "kurtosis_non_excess": float(stats.kurtosis(array, fisher=False, bias=False)),
        "zero_fraction": float(np.mean(array == 0.0)),
    }


def _worked_example() -> None:
    """Section 1: the hypothetical the checkpoint quotes, with its intermediates."""
    annual_sharpe, n_observations, n_trials = 2.0, 2520, 200
    skew, kurtosis = -0.5, 5.0
    daily_sharpe = annual_sharpe / math.sqrt(SESSIONS_PER_YEAR)
    standard_error = _sharpe_standard_error(daily_sharpe, n_observations, skew, kurtosis)
    psr = probabilistic_sharpe_ratio(daily_sharpe, 0.0, n_observations, skew, kurtosis)

    print("\n=== 1. WORKED EXAMPLE =========================================================")
    print(f"annual Sharpe {annual_sharpe}, n={n_observations} daily obs, trials={n_trials}, "
          f"skew={skew}, non-excess kurtosis={kurtosis}")
    print(f"daily Sharpe               = {daily_sharpe:.8f}")
    print(f"Sharpe standard error      = {standard_error:.8f} (daily units)")
    print(f"PSR vs zero (no deflation) = {psr:.8f}")
    print(f"{'V_annual':>10} {'V_daily':>12} {'E[max SR] daily':>17} {'E[max SR] annual':>18} "
          f"{'excess SR':>11} {'DSR':>12}")
    for annual_variance in (0.25, 1.00):
        daily_variance = annual_variance / SESSIONS_PER_YEAR
        threshold = expected_max_sharpe(n_trials, daily_variance)
        dsr = deflated_sharpe_ratio(
            observed_sharpe=daily_sharpe,
            n_trials=n_trials,
            n_observations=n_observations,
            skew=skew,
            kurtosis=kurtosis,
            variance_of_trial_sharpes=daily_variance,
        )
        print(f"{annual_variance:>10.2f} {daily_variance:>12.8f} {threshold:>17.8f} "
              f"{threshold * math.sqrt(SESSIONS_PER_YEAR):>18.6f} "
              f"{daily_sharpe - threshold:>11.8f} {dsr:>12.8f}")


def _sensitivity_table() -> None:
    """Section 2: the same strategy at rising trial counts, everything else held fixed."""
    daily_sharpe = 2.0 / math.sqrt(SESSIONS_PER_YEAR)
    common: dict[str, Any] = {
        "observed_sharpe": daily_sharpe,
        "n_observations": 2520,
        "skew": -0.5,
        "kurtosis": 5.0,
    }
    print("\n=== 2. DSR SENSITIVITY TO TRIAL COUNT ==========================================")
    print(f"{'N':>6} {'E[maxSR] ann (V=0.25)':>23} {'DSR (V=0.25)':>14} "
          f"{'E[maxSR] ann (V=1.00)':>23} {'DSR (V=1.00)':>14}")
    for n_trials in (1, 10, 50, 100, 200, 500, 1000):
        cells: list[str] = []
        for annual_variance in (0.25, 1.00):
            daily_variance = annual_variance / SESSIONS_PER_YEAR
            threshold = expected_max_sharpe(n_trials, daily_variance)
            dsr = deflated_sharpe_ratio(
                n_trials=n_trials, variance_of_trial_sharpes=daily_variance, **common
            )
            cells.append(f"{threshold * math.sqrt(SESSIONS_PER_YEAR):>23.6f} {dsr:>14.8f}")
        print(f"{n_trials:>6} " + " ".join(cells))


def _run(engine: BacktestEngine, strategy: Strategy, start: date, end: date) -> list[float]:
    """One backtest, returning the realised per-session net return series."""
    return engine.run(strategy, start, end).returns


def _control_window_comparison(
    engine: BacktestEngine, cfg: Config
) -> tuple[dict[str, float], dict[str, float]]:
    """Section 3a: the live control measured over 2023 alone and over the full dev window."""
    from tests.fixtures.clean.equal_weight_universe import EqualWeightUniverse

    short = _moments(_run(engine, EqualWeightUniverse(), CONTROL_START, CONTROL_END))
    full = _moments(_run(engine, EqualWeightUniverse(), cfg.dates.dev_start, cfg.dates.dev_end))

    print("\n=== 3. LIVE POSITIVE CONTROL: EqualWeightUniverse ==============================")
    # Two annualised Sharpes are printed on purpose. `metrics` is what src/eval/metrics.py reports
    # and what the rest of the project quotes: a geometric annualised return over a volatility
    # scaled by sqrt(250). `arith` is the per-observation Sharpe scaled by sqrt(252), which is the
    # quantity the DSR is defined on. They are not the same number and the gap is not rounding.
    print(f"{'window':<28} {'n':>6} {'Sharpe ann (metrics)':>21} {'Sharpe ann (arith)':>19} "
          f"{'Sharpe/obs':>12} {'skew':>8} {'kurt':>8}")
    for label, m in ((f"{CONTROL_START} .. {CONTROL_END}", short),
                     (f"{cfg.dates.dev_start} .. {cfg.dates.dev_end}", full)):
        print(f"{label:<28} {int(m['n']):>6} {m['sharpe_annualised_metrics']:>21.4f} "
              f"{m['sharpe_per_observation'] * math.sqrt(SESSIONS_PER_YEAR):>19.4f} "
              f"{m['sharpe_per_observation']:>12.6f} {m['skew']:>8.4f} "
              f"{m['kurtosis_non_excess']:>8.4f}")
    return short, full


def _deflate_control(
    moments: dict[str, float], trial_sharpes: list[float], corpus_size: int
) -> None:
    """Section 3b: put the 2023 figure through the deflation at several honest trial counts."""
    measured_variance = float(np.var(np.asarray(trial_sharpes), ddof=1))
    n_observations = int(moments["n"])
    psr = probabilistic_sharpe_ratio(
        moments["sharpe_per_observation"], 0.0, n_observations,
        moments["skew"], moments["kurtosis_non_excess"],
    )
    print(f"\ntrial-Sharpe dispersion measured across {corpus_size} clean fixtures on the same "
          f"window:")
    print(f"  V (per-observation units) = {measured_variance:.10f}  "
          f"-> annualised {measured_variance * SESSIONS_PER_YEAR:.6f}, "
          f"sd {math.sqrt(measured_variance * SESSIONS_PER_YEAR):.4f}")
    print(f"PSR of the 2023 series against zero, before any deflation: {psr:.6f} "
          f"(n={n_observations})")
    print(f"{'N':>6} {'E[maxSR] ann':>14} {'observed - threshold':>22} {'DSR':>12}")
    for n_trials in sorted({1, 10, corpus_size, 100, 200, 500}):
        threshold = expected_max_sharpe(n_trials, measured_variance)
        dsr = deflated_sharpe_ratio(
            observed_sharpe=moments["sharpe_per_observation"],
            n_trials=n_trials,
            n_observations=n_observations,
            skew=moments["skew"],
            kurtosis=moments["kurtosis_non_excess"],
            variance_of_trial_sharpes=measured_variance,
        )
        print(f"{n_trials:>6} {threshold * math.sqrt(SESSIONS_PER_YEAR):>14.6f} "
              f"{moments['sharpe_per_observation'] - threshold:>22.8f} {dsr:>12.8f}")


def _corpus_sharpes(
    engine: BacktestEngine, fixtures: list[tuple[str, Strategy]], start: date, end: date
) -> tuple[list[str], list[list[float]], list[dict[str, float]]]:
    """Run every fixture over one window; return names, return series, and their moments."""
    names: list[str] = []
    series: list[list[float]] = []
    moments: list[dict[str, float]] = []
    for name, strategy in fixtures:
        returns = _run(engine, strategy, start, end)
        names.append(name)
        series.append(returns)
        moments.append(_moments(returns))
    return names, series, moments


def _pbo_section(
    names: list[str], series: list[list[float]], moments: list[dict[str, float]]
) -> float:
    """Section 4: PBO over the fixture corpus, with the per-fixture inputs printed.

    Two sensitivities are reported alongside the headline. The split count is varied because
    tests/audit/test_stat.py records that a single PBO estimate carries a wide error bar, and a
    reader is entitled to see how much of the headline is the corpus and how much is the estimator.
    The second drops fixtures that hold nothing for most of the window — the panel begins mid-2019,
    so a 756-session lookback has no history to act on until 2022 and its flat stretch is an
    artefact of the data start rather than a strategy's behaviour.
    """
    lengths = {len(s) for s in series}
    if len(lengths) != 1:
        raise RuntimeError(f"fixture return series are not aligned: lengths {sorted(lengths)}")
    matrix = np.array(series, dtype=np.float64).T
    n_splits = 16
    pbo = probability_of_backtest_overfitting(matrix, n_splits=n_splits)

    print("\n=== 4. PROBABILITY OF BACKTEST OVERFITTING =====================================")
    print(f"{'fixture':<32} {'Sharpe ann':>12} {'Sharpe/obs':>12} {'zero-return frac':>18}")
    for name, m in zip(names, moments, strict=True):
        print(f"{name:<32} {m['sharpe_annualised_metrics']:>12.4f} "
              f"{m['sharpe_per_observation']:>12.6f} {m['zero_fraction']:>18.4f}")
    print(f"\nmatrix: {matrix.shape[0]} sessions x {matrix.shape[1]} strategies, "
          f"n_splits={n_splits}, partitions={math.comb(n_splits, n_splits // 2):,}")
    print(f"PBO = {pbo:.6f}")

    print("\nsensitivity to the split count (same corpus):")
    print(f"{'n_splits':>9} {'partitions':>12} {'PBO':>10}")
    for splits in (4, 8, 12, 16):
        print(f"{splits:>9} {math.comb(splits, splits // 2):>12,} "
              f"{probability_of_backtest_overfitting(matrix, n_splits=splits):>10.6f}")

    keep = [i for i, m in enumerate(moments) if m["zero_fraction"] <= MAX_ZERO_FRACTION]
    if len(keep) != len(names):
        dropped = [names[i] for i in range(len(names)) if i not in keep]
        trimmed = probability_of_backtest_overfitting(matrix[:, keep], n_splits=n_splits)
        print(f"\nsensitivity to mostly-flat series (dropping {dropped} for holding nothing on "
              f"more than {MAX_ZERO_FRACTION:.0%} of sessions):")
        print(f"  {len(keep)} strategies, n_splits={n_splits}, PBO = {trimmed:.6f}")
    return pbo


def _trial_counter_demo() -> None:
    """Section 5: record trials including failures, verify, then tamper and verify again."""
    print("\n=== 5. TRIAL COUNTER ===========================================================")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo_trial_ledger.jsonl"
        counter = TrialCounter(path)
        for name, outcome in (
            ("candidate_001", "evaluated"),
            ("candidate_002", "syntax_error"),
            ("candidate_003", "runtime_error"),
            ("candidate_004", "evaluated"),
            ("candidate_005", "syntax_error"),
        ):
            seq = counter.record(name, outcome)  # type: ignore[arg-type]
            print(f"  recorded seq={seq:<3} {name:<15} outcome={outcome}")
        print(f"  count()  = {counter.count()}  (failures included, which is the point)")
        print(f"  verify() = {counter.verify()}  entries verified")

        lines = path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[2])
        entry["outcome"] = "evaluated"          # relabel a failure as a success
        lines[2] = json.dumps(entry, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("  hand-edited line 3: outcome runtime_error -> evaluated, hash left untouched")
        print(f"  count()  = {counter.count()}  (counting is not verification)")
        try:
            counter.verify()
        except TamperError as exc:
            print(f"  verify() raised TamperError: {str(exc).split(': ', 1)[-1]}")
        else:
            raise RuntimeError("tampered ledger verified; the chain is not doing its job")


def _load_inputs(cfg: Config) -> tuple[TradingCalendar, pl.DataFrame, pl.DataFrame, list[Path]]:
    """Calendar, price panel and universe, plus the paths for the run manifest."""
    calendar_path = cfg.paths.data_raw / "calendar_cnx100.parquet"
    panel_path = cfg.paths.data_processed / "prices_adjusted.parquet"
    universe_path = cfg.paths.data_processed / "universe.parquet"
    return (
        load_calendar(calendar_path),
        pl.read_parquet(panel_path),
        pl.read_parquet(universe_path),
        [calendar_path, panel_path, universe_path],
    )


def main() -> int:
    cfg = load_config()
    seed_everything(cfg.meta.seed)
    _worked_example()
    _sensitivity_table()

    calendar, panel, universe, input_paths = _load_inputs(cfg)
    engine = BacktestEngine(panel, calendar, universe)
    fixtures = _load_clean_fixtures()

    with RunManifest(cfg, script="scripts/run_stat_audit.py") as run:
        for path in input_paths:
            run.add_input(path)

        control_short, control_full = _control_window_comparison(engine, cfg)
        # The dispersion of the search is measured on the same window the control is quoted over;
        # a threshold built from Sharpes measured elsewhere would not be comparable to it.
        _, _, control_moments = _corpus_sharpes(engine, fixtures, CONTROL_START, CONTROL_END)
        trial_sharpes = [m["sharpe_per_observation"] for m in control_moments]
        _deflate_control(control_short, trial_sharpes, len(fixtures))

        corpus = fixtures[:PBO_CORPUS_SIZE]
        names, series, moments = _corpus_sharpes(
            engine, corpus, cfg.dates.dev_start, cfg.dates.dev_end
        )
        pbo = _pbo_section(names, series, moments)

        _trial_counter_demo()

        run.note("control_sharpe_2023_annualised", control_short["sharpe_annualised_metrics"])
        run.note("control_sharpe_dev_annualised", control_full["sharpe_annualised_metrics"])
        run.note("pbo", pbo)
        run.note("pbo_corpus", names)
        run.note("clean_fixtures_run", len(fixtures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
