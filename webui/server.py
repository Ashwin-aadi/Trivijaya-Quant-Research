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

import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402
from run_corpus_backtest import _worker_init, run_one  # noqa: E402

from src.audit.stat import deflated_sharpe_ratio  # noqa: E402
from src.audit.static import Severity, audit_source  # noqa: E402
from src.capacity.deployability import (  # noqa: E402
    session_capacity,
    summarise_capacity,
    turnover_by_session,
)
from src.capacity.impact import add_daily_measures  # noqa: E402
from src.common.config import load_config  # noqa: E402
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
#: Net Sharpes seen this session. The deflation needs the spread of the trials searched over, and
#: estimating it from the search actually performed is the only honest source available here.
_SHARPES: list[float] = []
_LOCK = threading.Lock()


def _bump() -> int:
    """Count one evaluation, successful or not. Failures consumed search effort too."""
    global _TRIALS
    with _LOCK:
        _TRIALS += 1
        return _TRIALS


def _moments(returns: list[float]) -> tuple[float, float]:
    """Skew and kurtosis of a return series, computed rather than assumed normal."""
    n = len(returns)
    mean = sum(returns) / n
    centred = [r - mean for r in returns]
    sigma = (sum(c * c for c in centred) / n) ** 0.5
    if sigma <= 0:
        return 0.0, 3.0
    return (sum(c**3 for c in centred) / n / sigma**3,
            sum(c**4 for c in centred) / n / sigma**4)


def _variance_of_trials(sharpe: float) -> float | None:
    """Spread of this session's Sharpes, or None when one evaluation cannot show a spread.

    Returning None rather than a placeholder is deliberate. A made-up variance produces a deflated
    Sharpe that looks like a measurement, and on a single trial it produces a very flattering one --
    which is precisely the failure this repository exists to detect.
    """
    with _LOCK:
        _SHARPES.append(sharpe)
        if len(_SHARPES) < 2:
            return None
        mean = sum(_SHARPES) / len(_SHARPES)
        return sum((s - mean) ** 2 for s in _SHARPES) / (len(_SHARPES) - 1)


#: Loaded once at startup. Both are read-only inputs to the frozen benchmarks, and rebuilding the
#: liquidity frame per submission would add seconds to every run for an identical answer.
_LABELS: pl.DataFrame | None = None
_LIQUIDITY: pl.DataFrame | None = None
_CFG: Any = None


def _load_benchmark_inputs() -> None:
    """Regime labels for stage 2 and trailing liquidity for stage 3."""
    global _LABELS, _LIQUIDITY, _CFG  # noqa: PLW0603
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
    return {
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


def audit(source: str) -> dict[str, Any]:
    """Static leakage findings for pasted source: the whole list, not merely a verdict."""
    findings = audit_source(source, filename="submission.py")
    return {
        "rejected": any(f.severity is Severity.HIGH for f in findings),
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
        variance = _variance_of_trials(sharpe)
        if variance is not None:
            result["deflated_sharpe_probability"] = float(
                deflated_sharpe_ratio(sharpe, n_trials=trials,
                                      n_observations=result.get("n_sessions") or 1,
                                      skew=skew, kurtosis=kurtosis,
                                      variance_of_trial_sharpes=variance)
            )
            result["deflation_note"] = None
        result["skew"], result["kurtosis"] = skew, kurtosis
    result["session_trials"] = trials
    return result


class Server(ThreadingHTTPServer):
    """A second instance must fail to start rather than quietly share the port.

    ``HTTPServer`` sets ``allow_reuse_address``, and on Windows that lets two processes bind the
    same loopback port at once: both listen, requests are split between them unpredictably, and an
    edited server appears to have no effect because half the answers come from the old one. That
    cost an hour. There is no reason to reuse the address here, and failing loudly is the point.
    """

    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    """Two POST endpoints and the page itself. No routing framework; there are three routes."""

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
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        target = (STATIC / name).resolve()
        inside = target.is_relative_to(STATIC)
        if not inside or target.suffix not in SERVABLE or not target.is_file():
            self._send(b"not found", "text/plain; charset=utf-8", 404)
            return
        self._send(target.read_bytes(), f"{SERVABLE[target.suffix]}; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0]
        if route not in ("/api/audit", "/api/backtest"):
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
        payload: dict[str, Any] = audit(source) if route == "/api/audit" else backtest(source)
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
