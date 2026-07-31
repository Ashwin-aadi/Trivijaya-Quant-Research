"""The gross-to-net table: what transaction costs did to the corpus.

The PI's instruction at Checkpoint 1.1 was to report gross, net, and the difference, on the grounds
that the collapse between them is itself the empirical result — it is the demonstration of why a
backtest without a cost model is not evidence of anything.

Both figures come from a **single run**: the engine records the pre-cost return series alongside
the post-cost one, so the difference is a measured quantity on one path rather than a comparison of
two runs that could differ for some other reason.

Sign convention throughout: a *drag* is reported as the signed change (net minus gross), so it is
negative when costs hurt. Costs cannot help, and a positive drag anywhere is a bug, not a finding.

Usage:
    python scripts/gross_vs_net.py --results runs/pooled/backtest_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

#: Below this in absolute Sharpe a candidate never really traded. Such candidates are counted in
#: the corpus but excluded from drag statistics, because a strategy that holds cash has no cost to
#: report and averaging zeros into the drag would understate it.
FLAT_TOLERANCE = 1e-9
SESSIONS_PER_YEAR = 252.0


def _fmt(value: float | None, width: int = 8, places: int = 4) -> str:
    return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every number the checkpoint asks for, each with the sample it was computed over."""
    evaluated = [r for r in records if r["outcome"] == "evaluated"]
    traded = [
        r for r in evaluated
        if r.get("sharpe") is not None and r.get("sharpe_gross") is not None
        and abs(float(r["sharpe_gross"])) >= FLAT_TOLERANCE
    ]
    flat = [r for r in evaluated if r not in traded]

    sharpe_drop = [float(r["sharpe_gross"]) - float(r["sharpe"]) for r in traded]
    cagr_drop = [
        float(r["annualised_return_gross"]) - float(r["annualised_return"])
        for r in traded if r.get("annualised_return_gross") is not None
    ]
    # Turnover is recorded per session as a fraction of equity; annualising makes it comparable to
    # the way a fund would quote it, and makes the cost drag legible as turnover x rate.
    turnover_annual = [float(r["mean_turnover"]) * SESSIONS_PER_YEAR for r in traded]
    cost_annual = [float(r["mean_cost"]) * SESSIONS_PER_YEAR for r in traded]

    flipped = [r for r in traded if float(r["sharpe_gross"]) > 0 >= float(r["sharpe"])]
    ruined = [r for r in evaluated if r.get("ruined_on")]

    # Sign reversals, both directions. The negative-to-positive count is the diagnostic: costs
    # cannot improve a strategy, so anything other than zero here is a wiring fault rather than a
    # finding, and it is reported even though it should always be empty.
    to_negative = [r for r in traded if float(r["sharpe_gross"]) > 0 > float(r["sharpe"])]
    to_positive = [r for r in traded if float(r["sharpe_gross"]) < 0 < float(r["sharpe"])]

    return {
        "n_records": len(records),
        "n_evaluated": len(evaluated),
        "n_traded": len(traded),
        "n_flat": len(flat),
        "n_flipped": len(flipped),
        "share_flipped": len(flipped) / len(traded) if traded else None,
        "n_ruined": len(ruined),
        "n_sign_positive_to_negative": len(to_negative),
        "share_sign_positive_to_negative": len(to_negative) / len(traded) if traded else None,
        "n_sign_negative_to_positive": len(to_positive),
        "mean_sharpe_drop": statistics.mean(sharpe_drop) if sharpe_drop else None,
        "median_sharpe_drop": statistics.median(sharpe_drop) if sharpe_drop else None,
        "mean_cagr_drop": statistics.mean(cagr_drop) if cagr_drop else None,
        "mean_turnover_annual": statistics.mean(turnover_annual) if turnover_annual else None,
        "median_turnover_annual": statistics.median(turnover_annual) if turnover_annual else None,
        "mean_cost_annual": statistics.mean(cost_annual) if cost_annual else None,
        "median_cost_annual": statistics.median(cost_annual) if cost_annual else None,
        "best_gross": max((float(r["sharpe_gross"]) for r in traded), default=None),
        "best_net": max((float(r["sharpe"]) for r in traded), default=None),
        "flipped_names": sorted(r["name"] for r in flipped)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("runs/pooled/backtest_results.json"))
    parser.add_argument("--top", type=int, default=15, help="rows in the per-candidate table")
    args = parser.parse_args()

    records = json.loads(args.results.read_text(encoding="utf-8"))
    if not records or "sharpe_gross" not in records[0]:
        print(f"{args.results} predates the cost model and has no gross column; re-run the "
              "backtest before asking for this table.")
        return 1

    stats = summarise(records)

    print(f"\nGROSS vs NET  —  {args.results}")
    print(f"  candidates in file          {stats['n_records']}")
    print(f"  evaluated                   {stats['n_evaluated']}")
    print(f"  of which flat (never traded){stats['n_flat']:>6}")
    print(f"  of which traded             {stats['n_traded']:>6}   <- drag statistics use these")
    print()
    print(f"  best Sharpe gross           {_fmt(stats['best_gross'])}")
    print(f"  best Sharpe net             {_fmt(stats['best_net'])}")
    print(f"  mean Sharpe reduction       {_fmt(stats['mean_sharpe_drop'])}"
          f"   median {_fmt(stats['median_sharpe_drop'], 0)}")
    print(f"  mean CAGR reduction         {_fmt(stats['mean_cagr_drop'])}")
    print(f"  mean turnover, annualised   {_fmt(stats['mean_turnover_annual'])}"
          f"   median {_fmt(stats['median_turnover_annual'], 0)}")
    print(f"  mean cost drag, annualised  {_fmt(stats['mean_cost_annual'])}"
          f"   median {_fmt(stats['median_cost_annual'], 0)}")
    print()
    share = stats["share_flipped"]
    print(f"  profitable gross, unprofitable net   {stats['n_flipped']} of {stats['n_traded']}"
          f"  ({share:.1%})" if share is not None else "  none traded")
    print(f"  ruined outright by costs             {stats['n_ruined']}")
    sign_share = stats["share_sign_positive_to_negative"]
    print(f"  Sharpe sign reversed, positive -> negative   "
          f"{stats['n_sign_positive_to_negative']} of {stats['n_traded']}"
          + (f"  ({sign_share:.1%})" if sign_share is not None else ""))
    print(f"  Sharpe sign reversed, negative -> positive   "
          f"{stats['n_sign_negative_to_positive']}   (must be 0; costs cannot help)")

    traded = [r for r in records if r["outcome"] == "evaluated"
              and r.get("sharpe_gross") is not None
              and abs(float(r["sharpe_gross"])) >= FLAT_TOLERANCE]
    traded.sort(key=lambda r: -float(r["sharpe_gross"]))
    print(f"\n  top {args.top} by GROSS Sharpe, showing what costs did to each:")
    print(f"  {'candidate':<28} {'gross':>8} {'net':>8} {'drag':>8} {'turn/yr':>8} {'cost/yr':>8}")
    for record in traded[:args.top]:
        print(f"  {record['name']:<28} {_fmt(record['sharpe_gross'])} {_fmt(record['sharpe'])} "
              f"{_fmt(float(record['sharpe']) - float(record['sharpe_gross']))} "
              f"{_fmt(float(record['mean_turnover']) * SESSIONS_PER_YEAR)} "
              f"{_fmt(float(record['mean_cost']) * SESSIONS_PER_YEAR)}")

    out = args.results.parent / "gross_vs_net.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if stats["mean_sharpe_drop"] is not None and stats["mean_sharpe_drop"] < 0:
        print("\nHALT: mean Sharpe rose after costs were applied. Costs cannot improve a result; "
              "this is a sign error or a wiring fault, not a finding.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
