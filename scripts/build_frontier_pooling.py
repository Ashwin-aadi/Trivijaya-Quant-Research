"""Record which request each strategy in a frontier arm came from, and why that matters.

The frontier arms are collected as four independent chat requests of five strategies each, not as
twenty independent draws. Two strategies from the same reply are not independent observations: the
model wrote them in one pass, with the earlier ones in its context while it produced the later ones.
Any statistic that treats the arm as n=20 independent draws --- a binomial interval on a pass rate,
a duplicate count, a variance --- is therefore reported against a design this file describes rather
than against twenty free draws, and the study must say so.

This is provenance, not measurement. It reads the per-reply ``extraction.json`` files the extractor
already wrote and joins them to the flattened corpus, so the mapping from a ``<arm>_NNN.py`` file
back to the reply and the fenced block it came from survives after the flattening has happened.

Usage:
    python scripts/build_frontier_pooling.py --arm claude --requests 001 002 003 004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]

#: Why every arm is renamed away from ``candidate_NNN``. Kept verbatim in each arm's record because
#: the hazard it describes is silent: a parquet write does not warn about the file it replaces.
NAMESPACE_NOTE = (
    "Renamed candidate_NNN -> {arm}_NNN. data/interim/positions/ already held a "
    "candidate_019.parquet from the local P1 corpus; writing frontier positions under the original "
    "names would have overwritten a member of the frozen 156-strategy reference corpus."
)

POOLING_NOTE = (
    "{n} independent requests of five; draws within a request are not independent"
)

#: Spelled out for the counts an arm can plausibly have, so this script reproduces the record that
#: was written by hand for the first arm byte for byte. A regenerated artifact that differs from the
#: one it replaces only in prose is worse than useless: it makes a real change indistinguishable
#: from a cosmetic one in the diff.
_SPELLED = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def build(arm: str, requests: list[str]) -> dict[str, object]:
    """The flattened corpus, each entry tied back to its reply and block."""
    index: list[dict[str, object]] = []
    position = 0
    for request in requests:
        record_path = ROOT / "runs" / f"frontier_{arm}_{request}" / "extraction.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for block in record["blocks"]:
            index.append(
                {
                    "candidate": f"{arm}_{position:03d}.py",
                    "request": request,
                    "block": block["index"],
                    "heading": block["heading"],
                    "classes": block["strategy_classes"],
                }
            )
            position += 1
    return {
        "arm": arm,
        "requests": len(requests),
        "strategies": len(index),
        "note": POOLING_NOTE.format(n=_SPELLED.get(len(requests), len(requests))),
        "index": index,
        "namespace_note": NAMESPACE_NOTE.format(arm=arm),
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--requests", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None, help="defaults to the arm's run dir")
    args = parser.parse_args()

    payload = build(args.arm, args.requests)
    out = args.out or ROOT / "runs" / f"frontier_{args.arm}" / "pooling.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Verify the flattening this file describes actually matches what is on disk. If it does not,
    # every downstream join keyed on the candidate name is silently wrong.
    corpus = ROOT / "runs" / f"frontier_{args.arm}" / "candidates"
    on_disk = sorted(p.name for p in corpus.glob("*.py") if p.stem != "__init__")
    claimed = sorted(str(e["candidate"]) for e in payload["index"])  # type: ignore[index]
    if on_disk != claimed:
        _log.error("pooling index does not match the corpus on disk: %d vs %d files",
                   len(claimed), len(on_disk))
        return 1

    # Not ``relative_to``: ``--out`` may legitimately point outside the repository, and raising
    # there would fail the run after the artifact had already been written.
    shown = str(out.resolve()).replace(str(ROOT) + "\\", "").replace("\\", "/")
    _log.info("arm %s: %d strategies across %d requests -> %s",
              args.arm, payload["strategies"], payload["requests"], shown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
