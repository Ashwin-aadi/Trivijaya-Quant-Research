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
        # Read inside the block: run_one writes the series under tmp, which is about to vanish.
        if result.get("returns_path"):
            returns = pl.read_parquet(result["returns_path"])["return"].to_list()

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
    print("loading the development price panel (a few seconds)...", flush=True)
    try:
        # holdout=False is the default and there is no path here that changes it
        _worker_init()
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
        server = ThreadingHTTPServer((HOST, PORT), Handler)
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
