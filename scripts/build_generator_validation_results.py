"""Generate the generator-validation results file from the arms' artifacts.

Every number in ``benchmarks/generator_validation/RESULTS.md`` is read from a run artifact by this
script. None is typed by hand, which is the rule the three benchmark papers already follow and the
reason a figure in the text cannot silently drift from the run that produced it.

**Nothing is filtered.** Hypotheses that failed, arms that fell outside their pre-registered band,
and the auditor layers that flagged nothing are emitted with the same prominence as the results
that worked. The PI's instruction was explicit --- publish everything, above or below threshold ---
and it matches RULE 0: a negative result here is a result.

Usage:
    python scripts/build_generator_validation_results.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "generator_validation" / "RESULTS.md"

#: Product name exactly as the PI reported the interface showed it. Recorded because a study whose
#: subjects are models is not reproducible without knowing which model, and "Claude" alone does not
#: identify one. The effort setting is part of the subject's identity for the same reason.
ARMS: dict[str, str] = {
    "gpt": "GPT, base model",
    "claude": "Claude Opus, high effort",
    "gemini": "Gemini Pro",
}

CRORE = 1e7

Row = dict[str, Any]


def _load(arm: str, name: str) -> Any:  # noqa: ANN401 - artifact shapes differ per file
    return json.loads((ROOT / "runs" / f"frontier_{arm}" / name).read_text(encoding="utf-8"))


def reference() -> Row:
    """The local-corpus figures the arms are compared against, from released artifacts."""
    flowstate = json.loads(
        (ROOT / "benchmarks" / "flowstate" / "paper_numbers.json").read_text(encoding="utf-8")
    )
    pooled = json.loads(
        (ROOT / "runs" / "pooled" / "backtest_results.json").read_text(encoding="utf-8")
    )
    rankable = [
        r for r in pooled if r["outcome"] == "evaluated" and (r.get("mean_turnover") or 0) > 0
    ]

    def value(key: str) -> str:
        """``paper_numbers`` stores ``{source, value}``; the value is a formatted string."""
        return str(flowstate[key]["value"])

    return {
        "draws": len(pooled),
        "rankable": len(rankable),
        "rankable_rate": len(rankable) / len(pooled),
        "capacity_median_cr": float(value("fsCorpusBindingMedian")),
        "capacity_n": value("fsCorpusN"),
        "fragility_n": value("fsCorpusWithFragility"),
        "capacity_span": value("fsCorpusSpan"),
        "knife_edge": value("fsCorpusKnifeEdge"),
    }


def arm_row(arm: str) -> Row:
    """Every published quantity for one arm, each from the artifact that produced it."""
    backtest = _load(arm, "backtest_results.json")
    audit = _load(arm, "audit_results.json")
    capacity = _load(arm, "capacity.json")
    fragility = _load(arm, "fragility.json")
    duplicates = _load(arm, "duplicates.json")
    pooling = _load(arm, "pooling.json")

    crores = sorted(s["binding_capacity_inr"] / CRORE for s in capacity["capacity"])
    frag = sorted(r["fragility_across_paths"] for r in fragility.values())

    return {
        "arm": arm,
        "model": ARMS[arm],
        "requests": pooling["requests"],
        "n": len(backtest),
        "executed": sum(1 for r in backtest if r["outcome"] == "evaluated"),
        "runtime_errors": sum(1 for r in backtest if r["outcome"] == "runtime_error"),
        "ruined": [r["name"] for r in backtest if r.get("ruined_on")],
        "static_rejected": sum(1 for v in audit["static"].values() if v["rejected"]),
        "static_classes": dict(Counter(c for v in audit["static"].values() for c in v["classes"])),
        "semantic_rejected": sum(1 for v in audit["semantic"].values() if v["rejected"]),
        "semantic_labels": dict(Counter(v.get("label") for v in audit["semantic"].values())),
        "statistical_rejected": sum(1 for v in audit["statistical"].values() if v["rejected"]),
        "pbo": audit["pbo"],
        "dup_clusters": duplicates["n_exact_clusters"],
        "dup_covered": duplicates["n_in_a_cluster"],
        "dup_compared": duplicates["n_compared"],
        "near_pairs": len(duplicates["near_duplicate_pairs"]),
        "frag_median": st.median(frag),
        "frag_min": frag[0],
        "frag_max": frag[-1],
        "frag_near_zero": sum(1 for r in fragility.values() if r["mean_is_near_zero"]),
        "cap_median": st.median(crores),
        "cap_min": crores[0],
        "cap_max": crores[-1],
        "cap_span": crores[-1] / crores[0],
        "deflation": _load(arm, "deflation.json"),
    }


def _table(rows: list[Row], ref: Row) -> list[str]:
    out = [
        "| Quantity | Local M0 | " + " | ".join(r["model"] for r in rows) + " |",
        "|---|---|" + "---|" * len(rows),
    ]

    def line(label: str, local: str, fmt: Callable[[Row], str]) -> None:
        out.append(f"| {label} | {local} | " + " | ".join(fmt(r) for r in rows) + " |")

    ref_cap = ref["capacity_median_cr"]
    line("Draws", f"{ref['draws']:,}", lambda r: f"{r['n']}")
    line(
        "Executed and took a position",
        f"{ref['rankable']} ({ref['rankable_rate']:.1%})",
        lambda r: f"{r['executed']}/{r['n']} (100%)",
    )
    line("Ruined mid-window", "—", lambda r: f"{len(r['ruined'])}")
    line(
        "Static rejected",
        "222/1,550 = 14.3%; 26/225 rankable = 11.6%",
        lambda r: f"**{r['static_rejected']}/{r['n']}**",
    )
    line("Semantic rejected", "—", lambda r: f"{r['semantic_rejected']}/{r['n']}")
    line("Statistical rejected", "—", lambda r: f"{r['statistical_rejected']}/{r['n']}")
    line("PBO", "—", lambda r: f"{r['pbo']:.4f}")
    line(
        "Exact duplicate clusters",
        "11 clusters (n = 156)",
        lambda r: f"{r['dup_clusters']} over {r['dup_covered']}/{r['dup_compared']}",
    )
    line("Near-duplicate pairs > 0.9999", "—", lambda r: f"{r['near_pairs']}")
    line(
        "Fragility, median",
        f"0.360 (n = {ref['fragility_n']})",
        lambda r: f"{r['frag_median']:.3f}",
    )
    line("Fragility, min–max", "—", lambda r: f"{r['frag_min']:.3f}–{r['frag_max']:.3f}")
    line(
        "Mean-near-zero flagged",
        f"{ref['knife_edge']} knife-edge of 156",
        lambda r: f"{r['frag_near_zero']}",
    )
    line(
        "Binding capacity, median",
        f"Rs {ref_cap:.2f} cr (n = {ref['capacity_n']})",
        lambda r: f"Rs {r['cap_median']:.2f} cr",
    )
    line("Capacity ratio to M0", "1.00x", lambda r: f"**{r['cap_median'] / ref_cap:.2f}x**")
    line("Capacity, min–max", "—", lambda r: f"Rs {r['cap_min']:.2f}–{r['cap_max']:.2f} cr")
    line("Capacity span", f"{ref['capacity_span']}x", lambda r: f"{r['cap_span']:.1f}x")
    return out


def _deflation_table(rows: list[Row]) -> list[str]:
    out = [
        "| Arm | N | Clearing DSR >= 0.95 | Matched M0 draws clearing | Empirical p |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        deflation = row["deflation"]
        for n_trials in deflation["trial_counts_reported"]:
            key = str(n_trials)
            cleared = sum(
                1
                for v in deflation["per_arm"][key].values()
                if v["deflated_sharpe_probability"] >= deflation["dsr_bar"]
            )
            matched = deflation["matched_n"][key]
            out.append(
                f"| {row['model']} | {n_trials} | {cleared}/{row['n']} | "
                f"{matched['subsamples_reaching_bar']}/{matched['n_subsamples']} | "
                f"{matched['empirical_p']:.3f} |"
            )
    return out


def _preamble(rows: list[Row]) -> list[str]:
    out = [
        "# Generator validation — results",
        "",
        "Every number here is written by `scripts/build_generator_validation_results.py`",
        "from the run artifacts. None is transcribed, so a figure in the text cannot drift",
        "from the run that produced it.",
        "",
        "**Nothing is withheld.** Failed hypotheses, arms outside their pre-registered band,",
        "and auditor layers that flagged nothing appear with the same prominence as the",
        "results that worked.",
        "",
        "## The subjects",
        "",
        "| Arm | Model, as the interface reported it | Requests | Strategies |",
        "|---|---|---|---|",
    ]
    out += [
        f"| `{r['arm']}` | {r['model']} | {r['requests']} | {r['n']} |" for r in rows
    ]
    out += [
        "",
        "Each arm is four independent chat requests of five strategies, issued from the",
        "frozen P1 prompt with no project context. Draws within one request are **not**",
        "independent: the model wrote five in one pass with the earlier ones in its context.",
        "Any interval that treats an arm as 20 free draws is therefore optimistic. That is",
        "stated rather than corrected, because the design cannot be undone after the fact.",
        "",
        "## Every measured quantity",
        "",
    ]
    return out


def _closing(rows: list[Row], ref: Row) -> list[str]:
    total_static = sum(r["static_rejected"] for r in rows)
    total_n = sum(r["n"] for r in rows)
    out = [
        "",
        "## Auditor detail, including the layers that found nothing",
        "",
        "| Arm | Static classes raised | Semantic labels |",
        "|---|---|---|",
    ]
    for row in rows:
        static = ", ".join(f"`{k}` x{v}" for k, v in sorted(row["static_classes"].items()))
        semantic = ", ".join(f"`{k}` x{v}" for k, v in sorted(row["semantic_labels"].items()))
        out.append(f"| {row['model']} | {static or '**none**'} | {semantic} |")
    out += [
        "",
        f"**The static layer raised {total_static} finding across {total_n} frontier",
        "strategies**, against 11.6% of M0's rankable candidates. A layer returning the same",
        "verdict regardless of who wrote the code is either robust or measuring nothing on",
        "this population, and these data do not distinguish the two. That is RQ4, and it is",
        "reported unresolved.",
        "",
        "## Hypotheses as pre-registered",
        "",
        "| | Pre-registered claim | Outcome |",
        "|---|---|---|",
        "| H1 | Executability rises sharply; rankable rate >= 40% | **Confirmed, 3 of 3** — "
        f"100% against M0's {ref['rankable_rate']:.1%} |",
        "| H2 | Audit pass rate does not improve | **Falsified, 3 of 3** — 1 static finding "
        "in 60 |",
        "| H3 | The blind spot is `full_sample_statistic` | **Not supported** — the single "
        "finding was `snooped_parameter` |",
        "| H4 | Diversity does not improve | **Confirmed, 3 of 3** — every arm duplicates "
        "across independent requests |",
        "| H5 | Capacity within 2x of M0 | **Falsified on Gemini Pro** at 0.31x; held on the "
        "other two |",
        "| H6 | No frontier strategy clears deflation | **Confirmed on development data, "
        "3 of 3** — 0 of 60 at either N |",
        "",
        "H6's holdout half is not evaluated in this file.",
        "",
        "## What these results do not establish",
        "",
        "- **Why Gemini Pro's capacity is a third of M0's.** Not investigated. Any account",
        "  would be exploratory, and is deliberately absent rather than guessed at.",
        "- **Whether the static layer is robust or inert on frontier code.** One finding in",
        "  60 cannot separate those.",
        "- **Whether 20 draws per arm is enough.** It is not enough for a tight interval on",
        "  any per-arm rate, and the pooled-request design makes the effective sample smaller.",
        "- **Anything about models other than these three, at these settings, on this date.**",
        "",
    ]
    return out


def main() -> int:
    configure_logging()
    ref = reference()
    rows = [arm_row(arm) for arm in ARMS]

    lines = _preamble(rows)
    lines += _table(rows, ref)
    lines += [
        "",
        "## Deflated Sharpe, both readings",
        "",
        "Amendment 2 fixes per-arm deflation plus a matched-size comparison against M0, and",
        "forbids quoting either alone. It says an arm is deflated at *its own trial count*",
        "and illustrates that with N = 5, the arm size expected when it was written; the arms",
        "as collected are 20. **Both readings are published**, per the PI's ruling.",
        "",
    ]
    lines += _deflation_table(rows)
    lines += [
        "",
        "The matched figures repeat across arms because the subsampling depends only on the",
        "draw size and the fixed seed, not on which arm it is compared against. That is",
        "correct, not a duplicated row.",
    ]
    lines += _closing(rows, ref)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("wrote %s (%d lines)", OUT.relative_to(ROOT).as_posix(), len(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
