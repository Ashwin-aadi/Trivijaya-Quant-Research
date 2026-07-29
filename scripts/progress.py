"""Print how far the Phase 1.4 pipeline has got, from the artifacts on disk.

Reads state rather than tailing a log, so it is accurate even after a crash, a reboot, or a run
that was killed: whatever the files say is the truth. Safe to run at any time, including while a
stage is executing. It writes nothing.

Usage:
    python scripts/progress.py
    python scripts/progress.py --watch      # refresh every 10 seconds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

RUNS = Path("runs")

#: The two corpora that pool into the Phase 1.4 experiment, in generation order. They are held
#: separately rather than in one directory so each keeps its own generation summary and provenance;
#: the prompt digest is identical across both, which is what makes pooling legitimate.
CORPORA: tuple[tuple[str, Path, int], ...] = (
    ("batch 1", RUNS / "20260728T172115Z", 300),
    ("batch 2", RUNS / "batch2", 1250),
)


def bar(done: int, total: int, width: int = 34) -> str:
    """A fixed-width progress bar. Percentages alone are hard to read at a glance."""
    total = max(total, 1)
    filled = min(int(width * done / total), width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {done:>4}/{total} ({done / total:5.1%})"


def generation_eta(corpus: Path) -> str:
    """Remaining wall clock, from the rate the generator itself logged."""
    summary = corpus / "generation_summary.json"
    if not summary.exists():
        return ""
    data = json.loads(summary.read_text(encoding="utf-8"))
    drawn, requested = data.get("drawn", 0), data.get("requested", 0)
    seconds = data.get("wall_clock_seconds", 0.0)
    if drawn < 1 or drawn >= requested:
        return ""
    remaining = (seconds / drawn) * (requested - drawn) / 60
    return f"  ~{remaining:.0f} min left ({seconds / drawn:.0f}s each)"


def backtest_progress(corpus: Path) -> tuple[int | None, str]:
    """How far a running backtest has got, from the newest backtest log.

    The runner logs `N/M backtested` every 25 candidates. That counter counts *collected results*,
    successes and failures alike, which is the only thing that reflects progress through the corpus.
    Counting parquet files instead would be wrong: a re-run overwrites the same filenames for the
    same deterministically-successful candidates, so that count starts saturated at whatever the
    previous attempt left behind and never moves.
    """
    marker = str((corpus / "candidates").resolve())
    logs = [
        p for p in sorted(RUNS.glob("backtest_log*.txt"), key=lambda p: p.stat().st_mtime)
        if marker in p.read_text(encoding="utf-8", errors="replace")
    ]
    if not logs:
        return None, ""
    newest = logs[-1]
    last: int | None = None
    for line in newest.read_text(encoding="utf-8", errors="replace").splitlines():
        if "backtested" in line and "/" in line:
            fragment = line.split("|")[-1].strip().split()[0]
            head = fragment.split("/")[0]
            if head.isdigit():
                last = int(head)
    stamp = time.strftime("%H:%M:%S", time.localtime(newest.stat().st_mtime))
    return last, stamp


def corpus_report(label: str, corpus: Path, target: int) -> tuple[int, int, int]:
    """Print one corpus's stage progress, and return (executed, flat, rankable)."""
    print(f"\n{label}: {corpus.name}")
    if not (corpus / "candidates").is_dir():
        print("  not started")
        return 0, 0, 0

    generated = len(list((corpus / "candidates").glob("candidate_*.py")))
    print(f"  1 generate   {bar(generated, target)}{generation_eta(corpus)}")

    results_path = corpus / "backtest_results.json"
    if not results_path.exists():
        done, when = backtest_progress(corpus)
        state = "starting" if done is None else f"running - last update {when}"
        print(f"  2 backtest   {bar(done or 0, target)}  {state}")
        return 0, 0, 0

    results = json.loads(results_path.read_text(encoding="utf-8"))
    ran = sum(1 for r in results if r["outcome"] == "evaluated")
    flat = sum(
        1 for r in results
        if r["outcome"] == "evaluated" and r.get("sharpe") is not None
        and abs(float(r["sharpe"])) < 1e-9
    )
    print(f"  2 backtest   {bar(len(results), target)}  DONE")
    print(f"       executed {ran}/{len(results)}   of those, flat (never traded): {flat}")
    print(f"       rankable (executed and not flat): {ran - flat}")
    return ran, flat, ran - flat


def report() -> None:
    totals = [0, 0, 0]
    for label, corpus, target in CORPORA:
        for i, value in enumerate(corpus_report(label, corpus, target)):
            totals[i] += value

    executed, flat, rankable = totals
    print(f"\npooled: executed {executed}   flat {flat}   rankable {rankable}")
    # The abstention curve ranks on realised performance, so only the rankable set carries ordering
    # information. Below roughly 150 the curve cannot discriminate at low coverage and is not to be
    # plotted; that threshold was set before any of these numbers were seen.
    print(f"        AUAP power target 150 rankable: {'MET' if rankable >= 150 else 'not yet'}")

    audit_path = CORPORA[0][1] / "audit_results.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        print(
            f"  3 audit      static {len(audit.get('static', {}))}"
            f"  semantic {len(audit.get('semantic', {}))}"
            f"  statistical {len(audit.get('statistical', {}))}"
        )
    else:
        print("  3 audit      not started")

    ablation = [p for _, c, _ in CORPORA for p in c.glob("ablation_*.json")]
    print(f"  4 ablation   {'done: ' + ablation[0].name if ablation else 'not started'}")

    ledger = Path("data/processed/trial_ledger.jsonl")
    if ledger.exists():
        n = sum(1 for _ in ledger.open(encoding="utf-8"))
        # +10 for retries the first run recorded per candidate rather than per draw. The hash chain
        # was not edited to insert them; the correction is carried here and applied at deflation.
        print(f"\ntrial ledger: {n} recorded, N = {n + 10} for deflation (+10 known undercount)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="refresh every 10 seconds")
    args = parser.parse_args()

    if not args.watch:
        report()
        return 0
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear, then home
            print(time.strftime("%H:%M:%S"))
            report()
            time.sleep(10)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
