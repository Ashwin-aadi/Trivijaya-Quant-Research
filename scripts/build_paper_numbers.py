"""Emit every quantitative claim the RegimeStress write-up makes, straight from the artifacts.

Ordered by the PI on 2026-08-02, as a precondition of the benchmark freeze:

    "Before freezing the benchmark, verify that every quantitative claim in the paper is generated
    directly from the final artifacts and not copied manually from intermediate reports."

The AlphaAudit paper was written the other way round: its numbers were transcribed by hand from
committed checkpoint reports. That is auditable but not *mechanically* auditable — a transcription
error would look exactly like a result. Here the paper contains no figures at all. It cites LaTeX
macros, and this script is the only thing that defines them.

Three outputs, all regenerated together so they cannot drift apart:

* ``papers/regimestress_numbers.tex`` — ``\\newcommand`` definitions, one per claim.
* ``benchmarks/regimestress/paper_numbers.json`` — the same table as data, with the artifact each
  value came from, so a reader can check any single number without reading LaTeX.
* ``benchmarks/regimestress/RESULTS.md`` — rendered from ``RESULTS.template.md`` by substituting
  ``{{macro}}`` placeholders, so the benchmark's own results file is generated too.

One value is *computed* here rather than read: the duplicate-leakage delta. It is two
cross-validation fits on the persisted feature table, with and without the duplicate rows, at the
fixed seed. It is reproduced here rather than stored because the number is a *difference* between
two configurations of an experiment, and storing it would put a figure in the paper whose
provenance was a log line.

Usage:
    python scripts/build_paper_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.common.config import Config, load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.eval.numbers import (  # noqa: E402
    fixed,
    integer,
    macro_name,
    percent,
    plain,
    scientific,
    signed,
)
from src.stress.predictor import cross_validate  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "regimestress"
PAPERS = ROOT / "papers"
SEED = 42

#: The target the PI designated primary at Checkpoint 2.1: Tier 1, across synthetic paths.
PRIMARY_TARGET = "fragility_across_paths"
#: The specification that survives its own robustness checks. Reported alongside, never instead.
STABLE_TARGET = "fragility_across_regimes[log1p]"

#: All four target specifications carry macros, so the paper's tables are generated in full rather
#: than generated for the two the argument turns on and transcribed for the other two. A reader
#: comparing rows should be reading the same pipeline's output in every cell.
TARGET_PREFIX = {
    "fragility_across_paths[raw]": "PathsRaw",
    "fragility_across_paths[log1p]": "PathsLog",
    "fragility_across_regimes[raw]": "RegimesRaw",
    "fragility_across_regimes[log1p]": "RegimesLog",
}
#: Learning-curve sizes, spelled out because a LaTeX macro name may not contain a digit.
CURVE_WORDS = ("Forty", "Sixty", "Eighty", "Hundred")

Macros = dict[str, tuple[str, str]]  # name -> (rendered value, artifact it came from)


def _read(path: Path) -> dict[str, Any]:
    """Load one artifact. Every artifact this script reads is a JSON object at the top level."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object; the caller indexes it by key")
    return payload


def _put(macros: Macros, source: str, **values: str) -> None:
    """Record a batch of macros against the artifact they came from, refusing silent overwrites."""
    for name, rendered in values.items():
        macro_name(name)
        if name in macros:
            raise ValueError(f"macro {name!r} defined twice; the second value would win silently")
        macros[name] = (rendered, source)


# --- the population -------------------------------------------------------------------


def _population(processed: Path) -> Macros:
    """How many strategies entered, and how many survived each of the census filters."""
    macros: Macros = {}
    cal = _read(processed / "tier1_calibration.json")
    _put(
        macros, "data/processed/tier1_calibration.json",
        rsNCalibrated=integer(cal["n_strategies"]),
        rsNDeterministic=integer(cal["deterministic_strategies"]),
        rsNNondeterministic=integer(len(cal["nondeterministic"])),
        rsNRuntimeFailures=integer(len(cal["failures"])),
        rsCalibrationSecondsPerBacktest=fixed(cal["timing"]["mean_seconds_per_backtest"], 1),
        rsCalibrationProjectedMinutes=fixed(cal["timing"]["projected_minutes_100_paths"], 0),
    )
    excl = _read(BENCH / "excluded_nondeterministic.json")
    _put(
        macros, "benchmarks/regimestress/excluded_nondeterministic.json",
        rsNExcludedNondeterministic=integer(excl["n_excluded"]),
        rsNRetainedSurvivors=integer(excl["n_retained_survivors"]),
        rsNNondeterministicFactors=integer(excl["n_standard_factors_excluded"]),
        rsWorstNondeterministicSwing=fixed(
            max(row["max_abs_sharpe_swing"] for row in excl["excluded"]), 3
        ),
    )
    frag = _read(processed / "fragility.json")
    _put(
        macros, "data/processed/fragility.json",
        rsNStressed=integer(frag["n_strategies"]),
        rsNPrimary=integer(frag["n_primary"]),
        rsNKnifeEdge=integer(frag["n_knife_edge_excluded"]),
        rsNNearZeroMean=integer(frag["n_flagged_near_zero_mean"]),
        rsMedianFragilityRegimes=fixed(frag["median_across_regimes"], 3),
        rsIqrFragilityRegimesLow=fixed(frag["iqr_across_regimes"][0], 3),
        rsIqrFragilityRegimesHigh=fixed(frag["iqr_across_regimes"][1], 3),
    )
    return macros


def _regimes(processed: Path) -> Macros:
    """The causal labelling layer: how K was chosen, how much it moves, and what it agrees with."""
    macros: Macros = {}
    sel = _read(processed / "regime_k_selection.json")
    _put(
        macros, "data/processed/regime_k_selection.json",
        rsRegimeK=integer(sel["selected_k"]),
        rsRegimeSelectionSessions=integer(sel["selection_window"]["n_feature_rows"]),
        rsRegimeSelectionEnd=sel["selection_window"]["end"],
    )
    diag = _read(processed / "regime_diagnostics.json")
    occupancy = [diag["occupancy"][str(k)] for k in range(int(diag["k"]))]
    _put(
        macros, "data/processed/regime_diagnostics.json",
        rsNLabelledSessions=integer(diag["n_labelled_sessions"]),
        rsNRegimeRefits=integer(diag["n_refits"]),
        rsRegimeOccupancyMin=integer(min(occupancy)),
        rsRegimeOccupancyMax=integer(max(occupancy)),
        rsRegimeRevisionRate=percent(diag["stability"]["one_quarter_revision"]["pooled"]["rate"]),
        rsRegimeTerminalDisagreement=percent(
            diag["stability"]["terminal_disagreement"]["pooled"]["rate"]
        ),
        rsRegimeTrainingSessionsFirst=integer(diag["training_sessions_first_refit"]),
        rsRegimeTrainingSessionsLast=integer(diag["training_sessions_last_refit"]),
    )
    check = _read(processed / "burnin_cross_check.json")
    _put(
        macros, "data/processed/burnin_cross_check.json",
        rsBurninCompared=integer(check["compared"]),
        rsBurninBeyondTolerance=integer(check["n_beyond_tolerance"]),
        rsBurninWorstRelative=scientific(check["max_relative_difference"]),
    )
    events = pl.read_csv(BENCH / "sebi_events.csv")
    _put(
        macros, "benchmarks/regimestress/sebi_events.csv",
        rsNSebiEvents=integer(events.height),
    )
    return macros


def _resampler(processed: Path) -> Macros:
    """Counterfactual Regime Resampling: its one free parameter, and whether its output passes."""
    macros: Macros = {}
    cal = _read(processed / "crr_calibration.json")
    conditional, unconditional = cal["conditional_on_regime"], cal["unconditional"]
    failing = unconditional["outside"][0] if unconditional["outside"] else "none"
    percentile = next(
        (c["real_percentile"] for c in unconditional["comparisons"] if c["statistic"] == failing),
        float("nan"),
    )
    _put(
        macros, "data/processed/crr_calibration.json",
        rsBlockLength=fixed(cal["block_length"]["sessions"], 2),
        rsBlockBandwidth=integer(cal["block_length"]["bandwidth"]),
        rsCrrPaths=integer(cal["n_paths"]),
        rsCrrSessions=integer(cal["n_returns"]),
        rsCrrWindowStart=cal["window"]["start"],
        rsCrrWindowEnd=cal["window"]["end"],
        rsNMoments=integer(conditional["n_statistics"]),
        rsMomentsFailedConditional=integer(conditional["n_outside_95pc_interval"]),
        rsMomentsFailedUnconditional=integer(unconditional["n_outside_95pc_interval"]),
        rsFailingMomentPercentile=fixed(percentile, 1),
        rsCrrDrawSeconds=fixed(cal["draw_seconds_for_all_paths"], 3),
    )
    return macros


# --- the two tiers, and the shortcut that made the rerun unnecessary --------------------


def _tiers(processed: Path) -> Macros:
    """What the expensive experiment cost, and how far the cheap one reproduces it."""
    macros: Macros = {}
    paths = sorted((ROOT / "runs" / "tier1").glob("path_*.json"))
    seconds = [float(_read(p)["seconds"]) for p in paths]
    _put(
        macros, "runs/tier1/path_*.json",
        rsTierOnePaths=integer(len(seconds)),
        rsTierOneCpuHours=fixed(sum(seconds) / 3600.0, 1),
        rsTierOneSecondsPerPath=fixed(float(np.median(seconds)), 0),
    )
    comparison = _read(processed / "tier_comparison.json")
    agreement = comparison["tier_agreement"]["conditional"]
    _put(
        macros, "data/processed/tier_comparison.json",
        rsTierAgreementMeanSharpe=fixed(agreement["spearman_mean_sharpe"], 3),
        rsTierAgreementFragility=fixed(agreement["spearman_fragility"], 3),
        rsTierAgreementN=integer(agreement["n"]),
        rsTierMedianOne=fixed(agreement["median_tier1"], 3),
        rsTierMedianTwo=fixed(agreement["median_tier2"], 3),
        rsDefinitionAgreement=fixed(
            comparison["definition_agreement"]["conditional"]["spearman"], 3
        ),
        rsConditioningEffectPaths=fixed(
            comparison["conditioning_effect"]["fragility_across_paths"]["spearman"], 3
        ),
        rsConditioningEffectRegimes=fixed(
            comparison["conditioning_effect"]["fragility_across_regimes"]["spearman"], 3
        ),
    )
    tier2 = _read(processed / "tier2_fragility.json")
    _put(
        macros, "data/processed/tier2_fragility.json",
        rsTierTwoSeconds=fixed(tier2["wall_clock_seconds"], 1),
        rsTierTwoPaths=integer(tier2["n_paths"]),
    )
    return macros


def _shortcut(processed: Path) -> Macros:
    """The validated shortcut: bootstrapped fragility against the same quantity, computed free."""
    macros: Macros = {}
    decision = _read(processed / "rerun_decision.json")
    convergence = {int(k): v for k, v in decision["convergence"].items()}
    _put(
        macros, "data/processed/rerun_decision.json",
        rsShortcutSpearman=fixed(decision["real_vs_bootstrap_spearman"], 3),
        rsShortcutN=integer(decision["n_strategies"]),
        rsShortcutMedianReal=fixed(decision["median_real"], 3),
        rsShortcutMedianBootstrap=fixed(decision["median_bootstrap"], 3),
        rsConvergenceLow=fixed(convergence[min(convergence)], 3),
        rsConvergenceLowPaths=integer(min(convergence)),
        rsConvergenceHigh=fixed(convergence[max(convergence)], 3),
        rsConvergenceHighPaths=integer(max(convergence)),
        rsConvergenceAtHundred=fixed(convergence[100], 3),
    )
    return macros


# --- the exclusions, reported as numbers rather than as absences ------------------------


def _exclusions() -> Macros:
    """What the knife-edge and duplicate filters removed, and how the removed differ."""
    macros: Macros = {}
    knife = _read(BENCH / "knife_edge_stability.json")
    compare = knife["comparison"]
    _put(
        macros, "benchmarks/regimestress/knife_edge_stability.json",
        rsKnifeEdgeExcluded=integer(knife["n_excluded"]),
        rsKnifeEdgeRetained=integer(knife["n_retained"]),
        rsKnifeEdgeFactors=integer(knife["n_standard_factors_excluded"]),
        rsKnifeEdgeSwingMedian=fixed(knife["sharpe_swing"]["median"], 4),
        rsKnifeEdgeSwingMax=fixed(knife["sharpe_swing"]["max"], 3),
        rsKnifeEdgeSwingAboveHalf=integer(knife["sharpe_swing"]["n_above_0_5"]),
        rsKnifeEdgeTurnoverExcluded=fixed(compare["mean_turnover"]["excluded"]["median"], 3),
        rsKnifeEdgeTurnoverRetained=fixed(compare["mean_turnover"]["retained"]["median"], 3),
        rsKnifeEdgeHoldingExcluded=fixed(compare["mean_holding_period"]["excluded"]["median"], 1),
        rsKnifeEdgeHoldingRetained=fixed(compare["mean_holding_period"]["retained"]["median"], 1),
        rsKnifeEdgeHoldingsExcluded=fixed(compare["effective_holdings"]["excluded"]["median"], 3),
        rsKnifeEdgeHoldingsRetained=fixed(compare["effective_holdings"]["retained"]["median"], 3),
    )
    dup = _read(BENCH / "duplicates.json")
    _put(
        macros, "benchmarks/regimestress/duplicates.json",
        rsDuplicateClusters=integer(dup["n_clusters"]),
        rsDuplicateMembers=integer(dup["n_strategies_in_clusters"]),
        rsDuplicateRemoved=integer(dup["n_removed"]),
        rsDuplicateCompared=integer(dup["n_compared"]),
        rsDuplicateUncompared=integer(dup["n_uncompared_short_series"]),
        rsNearDuplicatePairs=integer(dup["n_near_duplicate_pairs_not_merged"]),
        rsNearDuplicateThreshold=fixed(dup["near_correlation_threshold"], 4),
        rsLargestDuplicateCluster=integer(max(len(c) for c in dup["clusters"])),
    )
    return macros


# --- the predictor, and the diagnosis of why it does not do better ----------------------


def _diagnosis(processed: Path) -> Macros:
    """Sample, conditioning, tail concentration, influence, learning curve, permutation test."""
    macros: Macros = {}
    diag = _read(processed / "predictor_diagnosis.json")
    collinearity = diag["collinearity"]
    _put(
        macros, "data/processed/predictor_diagnosis.json",
        rsPredictorRows=integer(diag["n_rows"]),
        rsPredictorFeatures=integer(diag["n_features"]),
        rsPermutations=integer(diag["permutations"]),
        rsFeatureCondition=fixed(collinearity["condition_number"], 1),
        rsFeatureWorstPairR=fixed(collinearity["max_abs_pairwise_correlation"], 3),
        rsFeatureWorstPairA=collinearity["worst_pair"][0].replace("_", r"\_"),
        rsFeatureWorstPairB=collinearity["worst_pair"][1].replace("_", r"\_"),
    )
    for label, prefix in TARGET_PREFIX.items():
        entry = diag["targets"][label]
        shape, perm = entry["shape"], entry["permutation_test"]
        curve = entry["learning_curve"]
        values = {
            f"rs{prefix}Skew": fixed(shape["skewness"], 2),
            f"rs{prefix}Kurtosis": fixed(shape["excess_kurtosis"], 1),
            f"rs{prefix}TailShareFive": percent(shape["variance_share_top_5"]),
            f"rs{prefix}TailShareTen": percent(shape["variance_share_top_10"]),
            f"rs{prefix}BestModel": entry["best_by_spearman"].replace("_", " "),
            f"rs{prefix}PermSpearman": signed(perm["kind_spearman"], 3),
            f"rs{prefix}PermNullSpearman": fixed(perm["null_abs_spearman_p95"], 3),
            f"rs{prefix}PermPSpearman": fixed(perm["p_value_spearman"], 3),
            f"rs{prefix}PermPRsquared": fixed(perm["p_value_r2"], 3),
            f"rs{prefix}InfluenceLargest": signed(entry["influence"]["largest_single_delta"], 3),
            f"rs{prefix}InfluenceWorst": entry["influence"]["most_influential"][0]["name"].replace(
                "_", r"\_"
            ),
        }
        for word, point in zip(CURVE_WORDS, curve, strict=True):
            values[f"rs{prefix}CurveAt{word}"] = signed(point["spearman_mean"], 3)
        _put(macros, f"data/processed/predictor_diagnosis.json :: {label}", **values)
    macros.update(_curve_sizes(diag))
    return macros


def _curve_sizes(diag: dict[str, Any]) -> Macros:
    """The learning-curve training-set sizes, emitted once because all four targets share them.

    Emitting one copy per target would have put the same four numbers in the macro table four
    times, and a table with redundant entries is one where a later divergence goes unnoticed. The
    agreement is asserted here instead, so a pipeline change that made the sizes differ per target
    fails loudly rather than silently publishing whichever copy the paper happened to cite.
    """
    curves = [diag["targets"][label]["learning_curve"] for label in TARGET_PREFIX]
    sizes = [[point["n"] for point in curve] for curve in curves]
    if any(other != sizes[0] for other in sizes[1:]):
        raise ValueError(f"learning-curve sizes differ between targets: {sizes}")
    macros: Macros = {}
    _put(
        macros, "data/processed/predictor_diagnosis.json :: learning curve sizes",
        **{
            f"rsCurveSizeAt{word}": integer(n)
            for word, n in zip(CURVE_WORDS, sizes[0], strict=True)
        },
    )
    return macros


#: The model ladder, lowest capacity first. The order is the argument: if the limitation were bias,
#: scores would rise along it.
MODEL_WORD = {
    "ridge": "Ridge", "lasso": "Lasso", "elastic_net": "ElasticNet",
    "random_forest": "Forest", "gradient_boosting": "Boosting",
}


def _model_cells(models: dict[str, Any], prefix: str) -> dict[str, str]:
    """Every cell of one target's model table, including the train-versus-out-of-fold gap."""
    values: dict[str, str] = {}
    for kind, word in MODEL_WORD.items():
        row = models[kind]
        values[f"rs{prefix}{word}Rsquared"] = signed(row["r2_model"], 3)
        values[f"rs{prefix}{word}TrainRsquared"] = signed(row["r2_train"], 3)
        values[f"rs{prefix}{word}Spearman"] = signed(row["spearman"], 3)
        values[f"rs{prefix}{word}Mae"] = fixed(row["mae_model"], 3)
        values[f"rs{prefix}{word}DropFive"] = signed(row["trimmed"]["5"]["r2"], 3)
        values[f"rs{prefix}{word}DropTen"] = signed(row["trimmed"]["10"]["r2"], 3)
        # The bias-versus-variance evidence: a model that underfits would show a small gap here.
        values[f"rs{prefix}{word}Gap"] = fixed(row["r2_train"] - row["r2_model"], 3)
    best = max(models, key=lambda k: models[k]["r2_model"])
    values[f"rs{prefix}BestRsquaredModel"] = best.replace("_", " ")
    values[f"rs{prefix}BestRsquared"] = signed(models[best]["r2_model"], 3)
    values[f"rs{prefix}BaselineMae"] = fixed(models[best]["mae_baseline"], 3)
    return values


def _models(processed: Path) -> Macros:
    """Five models on identical folds, and the bias-versus-variance gap that closes the question."""
    macros: Macros = {}
    diag = _read(processed / "predictor_diagnosis.json")
    for label, prefix in TARGET_PREFIX.items():
        _put(
            macros, f"data/processed/predictor_diagnosis.json :: {label}",
            **_model_cells(diag["targets"][label]["models"], prefix),
        )
    factors = _read(processed / "factor_design.json")
    _put(
        macros, "data/processed/factor_design.json",
        rsFactorCount=integer(factors["n_factors"]),
        rsFactorCondition=fixed(factors["condition_number"], 1),
        rsFactorWorstPair=fixed(factors["max_abs_pairwise_correlation"], 3),
    )
    return macros


def _importances(processed: Path) -> Macros:
    """The top permutation importances, which the deduplication changed completely."""
    macros: Macros = {}
    report = _read(processed / "fragility_predictor.json")
    primary = next(r for r in report["results"] if r["target"] == f"{PRIMARY_TARGET}[raw]")
    ranked = sorted(primary["importance"].items(), key=lambda kv: -kv[1])[:5]
    words = ("First", "Second", "Third", "Fourth", "Fifth")
    for word, (column, value) in zip(words, ranked, strict=True):
        _put(
            macros, "data/processed/fragility_predictor.json",
            **{
                f"rsImportance{word}Name": column.replace("_", r"\_"),
                f"rsImportance{word}Value": signed(value, 3),
            },
        )
    narratives = _read(processed / "stress_narratives.json")
    _put(
        macros, "data/processed/stress_narratives.json",
        rsNarrativeCount=integer(narratives["n"]),
        rsNarrativeSeconds=fixed(narratives["seconds_per_item"], 1),
        rsNarrativeModel=narratives["model_tag"].replace("_", r"\_"),
    )
    return macros


# --- the one number that is measured here rather than read ------------------------------


def _leakage(processed: Path) -> Macros:
    """Re-measure what the duplicate rows were worth, by fitting with and without them.

    Two fits of the same model on the same features at the same seed, differing only in whether the
    16 duplicate rows are present. The difference is the paper's claim about leakage, and it is a
    difference between experiments rather than a property of either, so it is computed rather than
    stored.
    """
    table = pl.read_parquet(processed / "characteristics.parquet")
    payload = _read(processed / "fragility.json")
    targets = pl.DataFrame([
        {"name": r["name"], PRIMARY_TARGET: r[PRIMARY_TARGET]} for r in payload["primary"]
    ])
    joined = table.join(targets, on="name", how="inner").filter(~pl.col("knife_edge")).sort("name")
    excluded = {"name", "knife_edge", "duplicate", "n_beta_sessions", PRIMARY_TARGET}
    columns = [
        c for c in joined.columns
        if c not in excluded and not c.endswith("_n")
        and not c.startswith("uni_") and joined[c].dtype.is_numeric()
    ]
    scores: dict[str, float] = {}
    for label, frame in (("with", joined), ("without", joined.filter(~pl.col("duplicate")))):
        result, _ = cross_validate(
            frame.select(columns).to_numpy().astype(float),
            frame[PRIMARY_TARGET].to_numpy().astype(float),
            frame["name"].to_list(),
            target_name=PRIMARY_TARGET, seed=SEED,
        )
        scores[label] = result.r2_model
        scores[f"{label}_n"] = float(result.n_rows)
    macros: Macros = {}
    _put(
        macros, "computed by scripts/build_paper_numbers.py from characteristics.parquet",
        rsLeakageWithDuplicates=signed(scores["with"], 3),
        rsLeakageWithoutDuplicates=signed(scores["without"], 3),
        rsLeakageDelta=fixed(scores["with"] - scores["without"], 3),
        rsLeakageNWith=integer(scores["with_n"]),
        rsLeakageNWithout=integer(scores["without_n"]),
    )
    return macros


# --- output ------------------------------------------------------------------------------


def _write_tex(macros: Macros, path: Path) -> None:
    lines = [
        "% Generated by scripts/build_paper_numbers.py. Do not edit.",
        "% Every quantitative claim in regimestress.tex resolves through one of these macros.",
        "% Editing this file by hand would defeat the only mechanism that keeps the paper",
        "% and the artifacts in agreement.",
        "",
    ]
    for name in sorted(macros):
        rendered, source = macros[name]
        lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}  % {source}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_results(macros: Macros, template: Path, out: Path) -> int:
    """Substitute ``{{macro}}`` placeholders, raising on any that has no definition."""
    text = template.read_text(encoding="utf-8")
    missing: list[str] = []
    for name in sorted(macros, key=len, reverse=True):
        text = text.replace("{{" + name + "}}", plain(macros[name][0]))
    for chunk in text.split("{{")[1:]:
        missing.append(chunk.split("}}")[0])
    if missing:
        raise ValueError(f"RESULTS.template.md refers to undefined macros: {sorted(set(missing))}")
    out.write_text(text, encoding="utf-8")
    return len(text.splitlines())


def _collect(cfg: Config) -> Macros:
    processed = cfg.paths.data_processed
    macros: Macros = {}
    for part in (
        _population(processed), _regimes(processed), _resampler(processed), _tiers(processed),
        _shortcut(processed), _exclusions(), _diagnosis(processed), _models(processed),
        _importances(processed), _leakage(processed),
    ):
        for name, value in part.items():
            if name in macros:
                raise ValueError(f"macro {name!r} defined by two collectors")
            macros[name] = value
    return macros


def main() -> int:
    configure_logging()
    cfg = load_config()
    with RunManifest(cfg, script="build_paper_numbers.py") as run:
        macros = _collect(cfg)
        _write_tex(macros, PAPERS / "regimestress_numbers.tex")
        (BENCH / "paper_numbers.json").write_text(
            json.dumps(
                {name: {"value": v, "source": s} for name, (v, s) in sorted(macros.items())},
                indent=2,
            ),
            encoding="utf-8",
        )
        template = BENCH / "RESULTS.template.md"
        rendered = (
            _render_results(macros, template, BENCH / "RESULTS.md") if template.exists() else 0
        )
        run.note("n_macros", len(macros))
        run.note("results_lines", rendered)
    _log.info("%d macros -> papers/regimestress_numbers.tex", len(macros))
    _log.info("%d distinct artifacts cited", len({s for _, s in macros.values()}))
    if rendered:
        _log.info("RESULTS.md rendered, %d lines", rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
