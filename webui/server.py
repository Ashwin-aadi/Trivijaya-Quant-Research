"""A local browser front end for auditing and backtesting one strategy at a time.

**What this is.** A convenience wrapper for someone who has cloned this repository and wants to try
a strategy of their own against the same auditor and the same backtester the papers used, without
learning the command line. It runs on their machine, against their clone, under their own user
account.

**What this is not.** It is not a service, and it is not Project 5. It accepts code from whoever is
sitting at the keyboard and executes it in this process with no sandbox, which is acceptable for
exactly one reason: the person submitting the code and the person running the server are the same
person, and they could have run `python` directly with less ceremony. The moment those two people
differ, that reasoning collapses and the sandbox requirements of CLAUDE.md Phase 5.0 apply in full.

So the listening socket is pinned to 127.0.0.1 and there is no flag to change it. Making this
reachable from another machine requires editing this file, which is the point: it should take a
deliberate act and a moment's thought, not a command-line argument typed in a hurry.

**Nothing here can affect a published result.** The holdout panel is never loaded -- the development
panel is a separate file and this process does not open the other one. The trial ledger is never
written; the session counter below lives in memory and dies with the process.

Usage:
    python webui/server.py                              # then open http://127.0.0.1:8000
    TRIVIJAYA_WEBUI_PORT=8010 python webui/server.py    # when 8000 is taken

The port is an environment variable because a clash is an inconvenience. The host is a constant in
this file because changing it is a decision, and a decision should be visible in a diff.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.stats as sps  # noqa: E402
from deduplicate_corpus import EXACT_TOLERANCE, NEAR_CORRELATION  # noqa: E402
from run_corpus_backtest import _worker_init, run_one  # noqa: E402

from src.audit import semantic as sem  # noqa: E402
from src.audit.stat import (  # noqa: E402
    TrialCounter,
    default_counter_path,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from src.audit.static import Severity, audit_source  # noqa: E402
from src.capacity.deployability import (  # noqa: E402
    capacity_by_flow_state,
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.impact import add_daily_measures  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.stress.characteristics import concentration, holding_period  # noqa: E402
from src.stress.fragility import across_regimes  # noqa: E402

#: Loopback only, deliberately not configurable. See the module docstring. The port is settable
#: because 8000 is a popular number and a collision is an inconvenience rather than a decision; the
#: host is not, because changing it is a decision and should read like one in a diff.
HOST = "127.0.0.1"
PORT = int(os.environ.get("TRIVIJAYA_WEBUI_PORT", "8000"))

STATIC = Path(__file__).resolve().parent
#: Served verbatim from this directory. Anything else is a 404, including a path that resolves
#: outside it -- a single-user loopback server makes that harmless, but "harmless" is not a reason
#: to hand out arbitrary files on disk.
SERVABLE = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}

#: Every strategy this session has evaluated. The Deflated Sharpe Ratio deflates by the number of
#: things tried, so a user who submits forty variants and keeps the best has earned a harsher
#: threshold than one who submits a single idea -- that is the whole lesson of P1, and it is the
#: reason this counter exists at all. It is per-session and in-memory: it never touches the
#: repository's tamper-evident ledger, which belongs to the published results.
_TRIALS = 0
#: One entry per trial that produced a return series, holding everything needed to deflate it
#: again later. Kept because N and the spread both grow as the session goes on, and a deflated
#: Sharpe computed against a smaller search is not comparable to one computed against a larger:
#: the whole ledger is recomputed on every run so the answer does not depend on paste order.
_SESSION_TRIALS: list[dict[str, Any]] = []
_LOCK = threading.Lock()

#: The engine reports an annualised Sharpe; every figure in ``src.audit.stat`` is per-observation at
#: the frequency of ``n_observations``, which here is daily sessions. Handing an annualised Sharpe
#: to those functions alongside a daily count is the mistake that module's docstring warns about by
#: name, and it inflates the answer badly -- the observed figure grows by sqrt(252) while the luck
#: threshold and the standard error do not follow it. 252 rather than the ~250 NSE trades, because
#: src/eval/metrics.py annualises on 252 and the two must agree or the conversion reintroduces the
#: mismatch it exists to remove.
SESSIONS_PER_YEAR = 252


def _bump() -> int:
    """Count one evaluation, successful or not. Failures consumed search effort too."""
    global _TRIALS
    with _LOCK:
        _TRIALS += 1
        return _TRIALS


def _moments(returns: list[float]) -> tuple[float, float]:
    """Skew and non-excess kurtosis of a return series, computed rather than assumed normal.

    Bias-corrected, and kurtosis on the non-excess convention, because that is exactly what
    ``scripts/deflate_standard_factors.py`` feeds the same frozen functions for the published
    figures. An earlier version here used the population estimators, which agreed to about three
    decimal places and moved the reported PSR in the fourth -- small, but it meant this page and the
    papers were answering the same question with two different conventions.
    """
    array = np.asarray(returns, dtype=float)
    if array.size < 2 or float(array.std(ddof=1)) <= 0:
        return 0.0, 3.0
    return float(sps.skew(array, bias=False)), float(sps.kurtosis(array, fisher=False, bias=False))


def _record_trial(index: int, label: str, sharpe_annual: float, daily: float, skew: float,
                  kurtosis: float, n_observations: int) -> None:
    """Keep what is needed to re-deflate this trial later, when N and the spread have grown.

    ``index`` is the number :func:`_bump` handed back for *this* request, not a re-read of the
    global. The server is threaded, so two runs started close together can otherwise both read the
    counter after both have incremented it and record themselves under the same number -- which was
    caught by running two strategies while a browser was working through a third.
    """
    with _LOCK:
        _SESSION_TRIALS.append({
            "index": index, "label": label, "sharpe_annual": sharpe_annual,
            "sharpe_per_observation": daily, "skew": skew, "kurtosis": kurtosis,
            "n_observations": n_observations,
        })


def _variance_of_trials() -> float | None:
    """Spread of this session's per-observation Sharpes, or None while one trial shows no spread.

    Per-observation, not annualised, so the figure is already in the units
    :func:`deflated_sharpe_ratio` expects. See :data:`SESSIONS_PER_YEAR`.

    Returning None rather than a placeholder is deliberate. A made-up variance produces a deflated
    Sharpe that looks like a measurement, and on a single trial it produces a very flattering one --
    which is precisely the failure this repository exists to detect.
    """
    with _LOCK:
        values = [float(t["sharpe_per_observation"]) for t in _SESSION_TRIALS]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((s - mean) ** 2 for s in values) / (len(values) - 1)


def session_ledger() -> dict[str, Any]:
    """Every trial this session, re-deflated at the *current* N and spread.

    This is how the corpus was deflated and the only way the numbers are comparable to each other:
    one N and one variance for the whole search, applied to every strategy in it. Deflating each
    submission against only the trials that happened to precede it would make the answer depend on
    the order you pasted things in, and a strategy is not more credible for having been typed first.

    Two different denominators, deliberately. **N counts every attempt**, including the ones that
    failed to run -- they consumed the search. **The spread comes only from trials that produced a
    return series**, because a strategy that never ran has no Sharpe to contribute. The corpus does
    exactly this: N = 1,887 from the ledger, variance from the 225 that were rankable.
    """
    variance = _variance_of_trials()
    with _LOCK:
        trials = list(_SESSION_TRIALS)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        row = dict(trial)
        row["deflated_sharpe_probability"] = None if variance is None else float(
            deflated_sharpe_ratio(
                float(trial["sharpe_per_observation"]), n_trials=_TRIALS,
                n_observations=int(trial["n_observations"]), skew=float(trial["skew"]),
                kurtosis=float(trial["kurtosis"]), variance_of_trial_sharpes=variance,
            )
        )
        rows.append(row)
    return {
        "n_trials": _TRIALS,
        "n_with_series": len(trials),
        "variance_of_trial_sharpes": variance,
        "luck_threshold_sharpe": None if variance is None else float(
            expected_max_sharpe(_TRIALS, variance)
        ),
        "rows": rows,
    }


#: Loaded once at startup. Both are read-only inputs to the frozen benchmarks, and rebuilding the
#: liquidity frame per submission would add seconds to every run for an identical answer.
_LABELS: pl.DataFrame | None = None
_LIQUIDITY: pl.DataFrame | None = None
_FLOWS: pl.DataFrame | None = None
_CFG: Any = None
#: Read once at startup and never written. This is the published corpus's tamper-evident count, and
#: it is shown beside the session counter so the session's N is never mistaken for the honest one.
_LEDGER: dict[str, Any] = {}
#: The published corpus's return series, and this session's, for the duplicate check.
_CORPUS: dict[str, np.ndarray] = {}
_CORPUS_DATES: list[Any] | None = None
#: (trial index, session dates, net returns) per evaluated submission. The index is carried
#: rather than inferred from position: a run that failed outright still consumed a trial number,
#: so list position and trial number diverge the moment anything goes wrong.
_SESSION_RETURNS: list[tuple[int, list[Any], np.ndarray]] = []


def _load_benchmark_inputs() -> None:
    """Regime labels for stage 2, trailing liquidity and flow states for stage 3."""
    global _LABELS, _LIQUIDITY, _FLOWS, _CFG  # noqa: PLW0603
    _CFG = load_config()
    _LABELS = pl.read_parquet(_CFG.paths.data_processed / "regime_labels.parquet").select(
        "session_date", "state"
    )
    panel = pl.read_parquet(_CFG.paths.data_processed / "prices_adjusted.parquet").filter(
        (pl.col("session_date") >= _CFG.dates.dev_start)
        & (pl.col("session_date") <= _CFG.dates.dev_end)
    )
    _LIQUIDITY = add_daily_measures(
        panel, adv_window=_CFG.constraints.adv_window_sessions
    ).select(["session_date", "symbol", "adv_inr"])
    _FLOWS = pl.read_parquet(_CFG.paths.data_processed / "participant_flows.parquet").select(
        ["session_date", "flow_state"]
    )
    _read_ledger()
    _load_corpus_returns()


def _load_corpus_returns() -> None:
    """The published corpus's realised net return series, for the duplicate check.

    Only strategies with the full session count are kept. A strategy ruined early has a shorter
    series and cannot be elementwise-compared with a complete one; treating a short series as a
    non-match would be right by accident, and padding it would manufacture agreement.
    """
    global _CORPUS_DATES  # noqa: PLW0603
    frame = pl.read_parquet(_CFG.paths.data_processed / "real_returns.parquet").select(
        ["name", "session_date", "net_return"]
    ).sort(["name", "session_date"])
    counts = frame.group_by("name").len()
    # The longest series present, which is the complete development window. Read through the frame
    # rather than assumed to be 1,232: the window is config, and a hardcoded length here would go
    # quietly wrong the day someone changes it.
    full = int(counts.select(pl.col("len").max()).item())
    for name, rows in frame.join(
        counts.filter(pl.col("len") == full).select("name"), on="name", how="inner"
    ).group_by("name"):
        ordered = rows.sort("session_date")
        if _CORPUS_DATES is None:
            _CORPUS_DATES = ordered["session_date"].to_list()
        _CORPUS[str(name[0])] = ordered["net_return"].to_numpy()


def duplicate_check(trial_index: int, dates: list[Any],
                    returns: np.ndarray) -> dict[str, Any]:
    """Is this strategy an earlier submission under a new name, and what is it closest to?

    Judged on the realised net return series and not on source text, at the tolerance the corpus
    census used: two strategies whose returns agree on every session are the same strategy for every
    purpose here, and two with near-identical code that diverge are not.

    **Two arms, and they are not equally strong.**

    *Within this session* the comparison is exact and sound: both series came out of this process,
    so identical returns mean an identical strategy.

    *Against the published corpus* only similarity is reported, never an exact verdict. Re-running
    ``tests/fixtures/clean/momentum_skip_month.py`` through this engine today reproduces
    ``standard_factor_deflation.json`` to the last decimal but correlates 0.9615 with the series
    stored under that name in ``real_returns.parquet`` -- and the *gross* series differ too, so the
    gap is in the positions rather than in costs. Until that is explained, an exact-match test
    against those series could only ever return "no", and reporting that as evidence of novelty
    would be a check that cannot fail pretending to be one that passed.

    **A duplicate still counts as a trial.** It consumed a search, and the deflated Sharpe is a
    statement about how many searches were made, not about how many distinct ideas they contained.
    """
    result: dict[str, Any] = {
        "available": True, "session_match": None, "nearest_name": None,
        "nearest_correlation": None, "tolerance": EXACT_TOLERANCE,
        "near_threshold": NEAR_CORRELATION, "n_corpus_compared": 0,
        "n_session_compared": 0, "corpus_exact_supported": False,
    }

    def correlation(series: np.ndarray) -> float | None:
        a, b = series - series.mean(), returns - returns.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denominator) if denominator > 0 else None

    best = -2.0
    if _CORPUS_DATES is not None and dates == _CORPUS_DATES:
        result["n_corpus_compared"] = len(_CORPUS)
        for name, series in _CORPUS.items():
            value = correlation(series)
            if value is not None and value > best:
                best, result["nearest_name"] = value, name
    if best > -2.0:
        result["nearest_correlation"] = best

    for index, previous_dates, previous in _SESSION_RETURNS:
        if previous_dates != dates:
            continue
        result["n_session_compared"] += 1
        if float(np.max(np.abs(previous - returns))) <= EXACT_TOLERANCE:
            result["session_match"] = result["session_match"] or f"submission #{index}"
    with _LOCK:
        _SESSION_RETURNS.append((trial_index, dates, returns))
    return result


def _read_ledger() -> None:
    """Verify and count the published trial ledger, read-only.

    Opened for reading and never for appending. The count belongs to the published corpus; a
    strategy tried at this console is not part of that search and must not inflate it, and the
    ledger is what every deflated Sharpe in the papers was computed against.
    """
    counter = TrialCounter(default_counter_path(_CFG))
    try:
        _LEDGER["verified"] = counter.verify()
        _LEDGER["intact"] = True
    except Exception as exc:  # noqa: BLE001 - a broken chain is a finding to display, not a crash
        _LEDGER["verified"] = counter.count()
        _LEDGER["intact"] = False
        _LEDGER["why"] = f"{type(exc).__name__}: {exc}"


def fragility(returns_frame: pl.DataFrame) -> dict[str, Any]:
    """RegimeStress stage 2, on the realised series: the charter's definition, no resampling.

    The synthetic-path variant is not run here. A thousand counterfactual histories per submission
    is a batch job, not something to make someone wait for at a keyboard, and the figure this
    returns is the one the published median is computed from.
    """
    assert _LABELS is not None
    joined = returns_frame.join(_LABELS, on="session_date", how="inner").sort("session_date")
    if joined.height < 2:
        return {"available": False, "why": "no sessions overlap the labelled regime window"}
    measured = across_regimes(
        "submission",
        joined["return"].to_numpy(),
        joined["state"].to_numpy(),
    )
    out = dict(measured.as_dict())
    out["available"] = True
    # Fragility is a ratio, and a strategy with performance near zero everywhere has a tiny
    # denominator. The frozen measure sets this flag rather than hiding the case.
    out["mean_is_near_zero"] = bool(measured.mean_is_near_zero)
    return out


def capacity(positions_path: str) -> dict[str, Any]:
    """FlowState stage 3: the largest book that stays inside the participation limit.

    A constraint computed from observed traded value, never an estimate of where an edge erodes.
    """
    assert _LIQUIDITY is not None and _CFG is not None
    weights = pl.read_parquet(positions_path).with_columns(factor=pl.lit("submission"))
    if weights.height == 0:
        # The frozen turnover routine builds a lagged frame by remapping session dates, and on an
        # empty input that column comes back typed Null and the join raises. Guard here rather than
        # in src/capacity/, which is frozen and correct for every book that has rows in it.
        return {"available": False, "why": "the strategy never held a position"}
    limit = _CFG.constraints.max_participation_rate
    traded = turnover_by_session(
        weights, min_traded_fraction=_CFG.constraints.min_traded_fraction
    )
    if traded.height == 0:
        return {"available": False, "why": "the book never changed by more than rounding noise"}
    per_session = session_capacity(traded, _LIQUIDITY, participation_limit=limit)
    if per_session.height == 0:
        return {"available": False, "why": "no traded name had a trailing liquidity figure"}
    summary = summarise_capacity(per_session, participation_limit=limit)[0]
    crore = 1e7
    # P3's actual contribution: whether deployable size collapses when foreign money leaves. The
    # session counts travel with the medians because the ratio between two states is the sentence a
    # reader will quote, and a ratio computed over thirty sessions is not evidence of anything.
    assert _FLOWS is not None
    by_state = [
        {"flow_state": row["flow_state"],
         "median_capacity_crore": row["median_capacity_inr"] / crore,
         "p05_capacity_crore": row["p05_capacity_inr"] / crore,
         "n_sessions": row["n_sessions"]}
        for row in capacity_by_flow_state(per_session, _FLOWS).to_dicts()
    ]
    return {
        "by_flow_state": by_state,
        "available": True,
        "binding_capacity_crore": summary.binding_capacity_inr / crore,
        "median_capacity_crore": summary.median_capacity_inr / crore,
        "entry_capacity_crore": summary.entry_capacity_inr / crore,
        "p05_capacity_crore": summary.p05_capacity_inr / crore,
        "participation_limit": limit,
        "n_rebalance_sessions": summary.n_rebalance_sessions,
        "fraction_bound_by_one_name": summary.fraction_bound_by_one_name,
        "binding_symbol": str(per_session.sort("capacity_inr").head(1)["binding_symbol"][0]),
    }


def book_measures(positions_path: str | None, sessions: list[Any]) -> dict[str, Any]:
    """Concentration and holding period from the position book, via the frozen P2 measures.

    These are the features the fragility predictor is built on, so they belong beside the fragility
    score rather than in a separate report. Sessions the book never appears in are rebuilt as empty
    dictionaries: a session holding nothing is a cash session, and dropping it would quietly raise
    every concentration figure by deleting the least concentrated days.
    """
    if positions_path is None:
        return {"available": False, "why": "the strategy held no position"}
    rows = pl.read_parquet(positions_path)
    held: dict[Any, dict[str, float]] = {}
    for row in rows.iter_rows(named=True):
        held.setdefault(row["session_date"], {})[row["symbol"]] = float(row["weight"])
    books = [held.get(session, {}) for session in sessions]
    return {"available": True, **concentration(books), **holding_period(books)}


def strategy_label(source: str, index: int) -> str:
    """The submitted class's own name, so the session ledger is readable at a glance."""
    try:
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                return node.name
    except SyntaxError:
        pass
    return f"submission #{index}"


def stated_rationale(tree: ast.Module) -> str:
    """The strategy's stated rationale, from wherever the author put it.

    Four places, in order: the module docstring, any bare string sitting at module level, the
    strategy class's docstring, and a class-level ``rationale = "..."`` constant. The generator was
    instructed to use the first. The console locks the import lines above the editor, which pushes
    a docstring typed at the top of the editable area *below* those imports -- where Python no
    longer calls it a docstring, though every human reader still would. Hence the second rule: a
    rationale the author plainly stated must not go unaudited on a technicality of placement.
    """
    module_doc = ast.get_docstring(tree)
    if module_doc:
        return module_doc
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return node.value.value
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_doc = ast.get_docstring(node)
        if class_doc:
            return class_doc
        for statement in node.body:
            if (isinstance(statement, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "rationale"
                            for t in statement.targets)):
                value = ast.literal_eval(statement.value) if isinstance(
                    statement.value, ast.Constant | ast.JoinedStr | ast.BinOp) else None
                if isinstance(value, str):
                    return value
    return ""


def semantic(source: str) -> dict[str, Any]:
    """A_sem: a local 7B reads the stated rationale against the code that implements it.

    A strategy with no rationale anywhere has stated no claim, and there is then nothing to check
    the code against -- that is reported as such rather than audited against an empty string, which
    would return a label about a claim nobody made.
    """
    try:
        rationale = stated_rationale(ast.parse(source))
    except (SyntaxError, ValueError) as exc:
        return {"available": False, "why": f"the source does not parse: {exc}"}
    if not rationale.strip():
        return {"available": False,
                "why": "no docstring and no rationale constant, so the strategy states no claim "
                       "for the model to check the code against"}
    if not sem.is_available():
        return {"available": False,
                "why": f"no Ollama server answering at {sem.OLLAMA_HOST}. Start it with `ollama "
                       f"serve`, then `ollama pull {sem.MODEL_TAG}`."}
    try:
        finding = sem.classify(rationale, source)
    except (sem.SemanticAuditUnavailable, sem.SemanticAuditParseError) as exc:
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "label": finding.label, "confidence": finding.confidence,
            "explanation": finding.explanation, "model_tag": finding.model_tag,
            "is_defect": finding.is_defect, "rationale": rationale.strip()}


def audit(source: str) -> dict[str, Any]:
    """Static leakage findings for pasted source: the whole list, not merely a verdict."""
    findings = audit_source(source, filename="submission.py")
    by_severity: dict[str, int] = {}
    for finding in findings:
        by_severity[finding.severity.value] = by_severity.get(finding.severity.value, 0) + 1
    return {
        "rejected": any(f.severity is Severity.HIGH for f in findings),
        "by_severity": by_severity,
        "findings": [
            {"leak_class": f.leak_class.value, "severity": f.severity.value, "line": f.line_number,
             "snippet": f.code_snippet, "explanation": f.explanation}
            for f in findings
        ],
    }


def backtest(source: str) -> dict[str, Any]:
    """Run pasted source through the real engine on the development panel, then deflate."""
    trials = _bump()
    returns: list[float] | None = None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "submission.py"
        path.write_text(source, encoding="utf-8")
        result = run_one(str(path), tmp)
        # Read inside the block: run_one writes both frames under tmp, which is about to vanish.
        if result.get("returns_path"):
            frame = pl.read_parquet(result["returns_path"])
            returns = frame["return"].to_list()
            # Stages 2 and 3 of the published funnel, on this one strategy, through the frozen
            # modules. Neither is allowed to take the whole run down: a benchmark that cannot be
            # computed says so, and the backtest above still stands on its own.
            try:
                result["fragility"] = fragility(frame.select("session_date", "return"))
            except Exception as exc:  # noqa: BLE001 - a stage failing is a datum, not a crash
                result["fragility"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}
            if result.get("positions_path"):
                try:
                    result["capacity"] = capacity(result["positions_path"])
                except Exception as exc:  # noqa: BLE001
                    result["capacity"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}
            else:
                result["capacity"] = {"available": False, "why": "the strategy held no position"}
            try:
                result["book"] = book_measures(
                    result.get("positions_path"), frame["session_date"].to_list()
                )
            except Exception as exc:  # noqa: BLE001
                result["book"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}
            try:
                result["duplicate"] = duplicate_check(
                    trials, frame["session_date"].to_list(), frame["return"].to_numpy()
                )
            except Exception as exc:  # noqa: BLE001
                result["duplicate"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}

    result["deflated_sharpe_probability"] = None
    result["deflation_note"] = (
        "Deflation needs at least two evaluations this session to estimate the spread of what you "
        "searched over. Run another strategy."
    )
    sharpe = result.get("sharpe")
    if result.get("outcome") == "evaluated" and sharpe is not None and returns:
        # Skew and kurtosis come from this strategy's own returns, never assumed normal: the whole
        # point of the Bailey-Lopez de Prado correction is that fat-tailed, negatively skewed
        # returns need more evidence to be believed, and passing 0 and 3 hands that back.
        skew, kurtosis = _moments(returns)
        observations = result.get("n_sessions") or 1
        # Per-observation, because n_observations is a count of daily sessions. Everything handed to
        # src.audit.stat from here down is daily; the annualised figure stays in the results table.
        daily = float(sharpe) / SESSIONS_PER_YEAR**0.5
        result["sharpe_per_observation"] = daily
        # PSR first, because it is the same statistic with the search term switched off: it asks
        # whether this Sharpe beats zero given the sample length and the shape of the returns. The
        # distance between it and the DSR below is precisely what the trial count costs.
        result["probabilistic_sharpe"] = float(
            probabilistic_sharpe_ratio(daily, benchmark_sharpe=0.0, n_observations=observations,
                                       skew=skew, kurtosis=kurtosis)
        )
        _record_trial(trials, strategy_label(source, trials), float(sharpe), daily, skew,
                      kurtosis, int(observations))
        variance = _variance_of_trials()
        if variance is not None:
            result["luck_threshold_sharpe"] = float(expected_max_sharpe(trials, variance))
            result["deflated_sharpe_probability"] = float(
                deflated_sharpe_ratio(daily, n_trials=trials,
                                      n_observations=observations,
                                      skew=skew, kurtosis=kurtosis,
                                      variance_of_trial_sharpes=variance)
            )
            result["deflation_note"] = None
        result["skew"], result["kurtosis"] = skew, kurtosis
    result["session_trials"] = trials
    result["ledger_trials"] = _LEDGER.get("verified")
    # The whole session re-deflated at the N this run just produced, so every earlier strategy is
    # judged against the same search as this one. See session_ledger() for why order must not count.
    result["session"] = session_ledger()
    return result


def status() -> dict[str, Any]:
    """What this console is measuring against, including the one stage it will not run.

    The holdout entry is a constant, not a switch. This process never opens the holdout panel and
    there is no request that makes it: under charter RULE 7 the holdout is evaluated once per
    project with logged PI authorisation, and a console that scored arbitrary pasted code against
    it -- unbounded, unlogged, and once per keystroke -- would consume the only clean data the
    repository has and take every published out-of-sample number down with it.
    """
    assert _CFG is not None
    return {
        "dev_start": str(_CFG.dates.dev_start),
        "dev_end": str(_CFG.dates.dev_end),
        "holdout_start": str(_CFG.dates.holdout_start),
        "holdout_end": str(_CFG.dates.holdout_end),
        "holdout_available": False,
        "holdout_reason": (
            "Evaluated three times under logged PI authorisation and closed. Nothing typed here "
            "can reopen it, and no result from this console is an out-of-sample result."
        ),
        "participation_limit": _CFG.constraints.max_participation_rate,
        "n_regimes": int(_LABELS.select(pl.col("state").n_unique()).item()) if _LABELS is not None
        else None,
        "semantic_model": sem.MODEL_TAG,
        "semantic_available": sem.is_available(),
        "ledger": _LEDGER,
        "session_trials": _TRIALS,
    }


class Server(ThreadingHTTPServer):
    """A second instance must fail to start rather than quietly share the port.

    ``HTTPServer`` sets ``allow_reuse_address``, and on Windows that lets two processes bind the
    same loopback port at once: both listen, requests are split between them unpredictably, and an
    edited server appears to have no effect because half the answers come from the old one. That
    cost an hour. There is no reason to reuse the address here, and failing loudly is the point.
    """

    allow_reuse_address = False


#: The three stages a submission can ask for, and nothing else. There is no holdout route: see
#: :func:`status`, where the reason is written down rather than left to be inferred from an absence.
ROUTES: dict[str, Callable[[str], dict[str, Any]]] = {
    "/api/audit": audit,
    "/api/semantic": semantic,
    "/api/backtest": backtest,
}


class Handler(BaseHTTPRequestHandler):
    """Three POST endpoints, a status endpoint, and the page itself. No routing framework."""

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Nothing here is worth caching, and a stale page after editing index.html is a confusing
        # ten minutes for whoever is trying to change it.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/api/status", "/api/session"):
            payload = status() if path == "/api/status" else session_ledger()
            self._send(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        target = (STATIC / name).resolve()
        inside = target.is_relative_to(STATIC)
        if not inside or target.suffix not in SERVABLE or not target.is_file():
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        self._send(target.read_bytes(), f"{SERVABLE[target.suffix]}; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in ROUTES:
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            source = json.loads(self.rfile.read(length) or b"{}").get("source", "")
        except json.JSONDecodeError as exc:
            self._send(json.dumps({"error": f"malformed request: {exc}"}).encode("utf-8"),
                       "application/json; charset=utf-8", 400)
            return
        if not isinstance(source, str) or not source.strip():
            self._send(json.dumps({"error": "no source submitted"}).encode("utf-8"),
                       "application/json; charset=utf-8", 400)
            return
        payload: dict[str, Any] = ROUTES[route](source)
        self._send(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        """Quieter than the default, which prints a line per asset request."""
        if args and "api" in str(args[0]):
            sys.stderr.write(f"{fmt % args}\n")


def main() -> int:
    print("loading the development price panel, regime labels and liquidity "
          "(this takes a few seconds)...", flush=True)
    try:
        # holdout=False is the default and there is no path here that changes it. Positions are
        # recorded because stage 3 measures capacity from the book, not from the return series.
        _worker_init(record_positions=True)
        _load_benchmark_inputs()
    except FileNotFoundError as exc:
        # The panel is derived, not shipped: a fresh clone has no data/ at all. Say which command
        # builds it rather than showing a traceback about a missing parquet file.
        print(f"\n  Could not load the price panel: {exc}\n\n"
              "  The panel is built from raw data, which this repository does not ship. Run:\n"
              "      python scripts/download_bhavcopy.py\n"
              "      python scripts/build_universe.py\n"
              "      python scripts/build_corporate_actions.py\n", file=sys.stderr)
        return 1

    try:
        server = Server((HOST, PORT), Handler)
    except OSError as exc:
        print(f"\n  Cannot listen on {HOST}:{PORT} -- {exc}\n\n"
              "  Something else is probably using that port. Pick another:\n"
              f"      TRIVIJAYA_WEBUI_PORT=8010 python webui/server.py\n", file=sys.stderr)
        return 1

    print(f"\n  Trivijaya-Quant strategy console -> http://{HOST}:{PORT}\n"
          f"  Loopback only. Ctrl-C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped. The session trial counter died with the process, as intended.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
