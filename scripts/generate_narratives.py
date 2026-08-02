"""Generate stress narratives for the most and least fragile strategies, after a throughput check.

RULE 5 requires a VRAM and throughput measurement before any batch inference, and a halt if the
estimate exceeds 30 minutes. The batch here is deliberately small — the extremes of the fragility
distribution, not the whole corpus — because the narratives exist to be *read* by the PI at
Checkpoint 2.2, and twelve is already more than anyone will read carefully.

Selection is by the primary target, ``fragility_across_regimes``, computed on the real series.
Knife-edge strategies are not narrated: a story about a strategy whose Sharpe moves by 2.5 when its
inputs move in the fifteenth decimal place would be a story about arithmetic.

Writes ``data/processed/stress_narratives.json``.

Usage:
    python scripts/generate_narratives.py --per-side 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from src.audit.semantic import MODEL_TAG, is_available, throughput_estimate  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402
from src.stress.narrative import build_facts, generate  # noqa: E402

_log = get_logger(__name__)

#: The charter's cap on an unreported step. Beyond this the run halts and asks.
MAX_UNREPORTED_SECONDS = 30 * 60


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-side", type=int, default=6,
                        help="how many most-fragile and least-fragile strategies to narrate")
    args = parser.parse_args()

    cfg = load_config()
    processed = cfg.paths.data_processed

    if not is_available():
        _log.error("Ollama is not reachable; start it with `ollama serve`. Nothing was generated.")
        return 1

    payload = json.loads((processed / "fragility.json").read_text(encoding="utf-8"))
    rows = [
        r for r in payload["primary"]
        if np.isfinite(r["fragility_across_regimes"]) and not r["mean_is_near_zero"]
    ]
    rows.sort(key=lambda r: r["fragility_across_regimes"])
    selected = rows[: args.per_side] + rows[-args.per_side :]
    _log.info(
        "%d strategies selected from %d rankable (fragility %.3f .. %.3f)",
        len(selected), len(rows),
        selected[0]["fragility_across_regimes"], selected[-1]["fragility_across_regimes"],
    )

    # RULE 5: measure before committing. The first call includes model load, and it is kept in the
    # average rather than discarded, which biases the estimate upward — the safe direction.
    estimate = throughput_estimate(len(selected), n_calls=3)
    _log.info(
        "throughput: %.1f s/item over %d timed calls -> %.1f min for %d items",
        estimate.seconds_per_item, len(estimate.call_seconds),
        estimate.estimated_seconds / 60, len(selected),
    )
    if estimate.estimated_seconds > MAX_UNREPORTED_SECONDS:
        _log.error(
            "estimated %.1f min exceeds the %d-minute cap; halting for a PI decision",
            estimate.estimated_seconds / 60, MAX_UNREPORTED_SECONDS // 60,
        )
        return 2

    characteristics = pl.read_parquet(processed / "characteristics.parquet")
    lookup = {row["name"]: row for row in characteristics.iter_rows(named=True)}

    started = time.perf_counter()
    with RunManifest(cfg, script="generate_narratives.py") as run:
        run.add_model(MODEL_TAG)
        narratives = []
        for position, record in enumerate(selected, start=1):
            name = record["name"]
            facts = build_facts(name, record, lookup.get(name, {}))
            result = generate(name, facts)
            entry = dict(result.as_dict())
            entry["group"] = "least_fragile" if position <= args.per_side else "most_fragile"
            narratives.append(entry)
            _log.info("  %2d/%d  %-18s %s", position, len(selected), name,
                      result.narrative[:90].replace("\n", " "))

        out = processed / "stress_narratives.json"
        out.write_text(
            json.dumps(
                {"model_tag": MODEL_TAG, "n": len(narratives),
                 "seconds_per_item": estimate.seconds_per_item, "narratives": narratives},
                indent=2, sort_keys=True,
            ),
            encoding="utf-8",
        )
        run.note("narratives", len(narratives))
        run.note("wall_clock_seconds", time.perf_counter() - started)

    unparsed = sum(1 for n in narratives if n["narrative"].startswith("["))
    _log.info("%d narratives in %.1f min (%d unparsed) -> %s",
              len(narratives), (time.perf_counter() - started) / 60, unparsed, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
