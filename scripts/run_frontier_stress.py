"""Run the frozen RegimeStress tier-1 suite over one frontier arm, changing only the population.

``run_stress_tier1.py`` is hardwired to ``benchmarks/alphaaudit/survivors`` --- correctly, since it
is the released P2 artifact and its population is part of what was frozen. The addendum needs the
identical measurement over a different set of strategies, so this script imports that module and
reuses its panel loader, its bootstrap draw, its worker initialiser and its per-path runner
unchanged. **The only thing it supplies of its own is the list of strategies.**

That is deliberate and it is the whole design of the addendum: the instrument is fixed and the
population is the variable. If any measurement code were copied here rather than imported, the two
arms would be measured by two implementations that could drift, and the comparison would be
meaningless the first time one of them was edited.

**Artifacts are namespaced by arm.** ``data/interim/positions/`` already contains a
``candidate_019.parquet`` belonging to the local P1 corpus, and the frontier arm's files were
originally named the same way. Writing them under the shared names would have overwritten a member
of the frozen 156-strategy reference corpus --- silently, since a parquet write does not complain
about a file it replaces. Everything this script produces therefore lives under
``runs/frontier_<arm>/``.

Usage:
    python scripts/run_frontier_stress.py --arm gpt --paths 100 --workers 24
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402

from run_stress_tier1 import (  # noqa: E402
    _dispatch,
    draw_paths,
    load_dev_panel,
)

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def arm_entries(arm: str) -> list[tuple[str, str]]:
    """``(name, source path)`` for every strategy in the arm, in stable order.

    No exclusion is applied here. P2 excludes nondeterministic and knife-edge strategies from its
    *statistics*, and that judgement is made after measurement from the measurement itself; applying
    it before would silently drop strategies the addendum is meant to count.
    """
    corpus = ROOT / "runs" / f"frontier_{arm}" / "candidates"
    paths = sorted(p for p in corpus.glob("*.py") if p.stem != "__init__")
    if not paths:
        raise SystemExit(f"no strategies found in {corpus}")
    return [(p.stem, str(p)) for p in paths]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, help="arm name, e.g. gpt")
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Pinned exactly as the frozen runner pins them. A different hash seed or thread count would
    # make this arm incomparable with the reference corpus for reasons having nothing to do with
    # the generator.
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"

    cfg = load_config()
    panel, _ = load_dev_panel(cfg)
    sessions = sorted(panel["session_date"].unique().to_list())
    paths, block_length = draw_paths(cfg, sessions, args.paths)
    entries = arm_entries(args.arm)

    out_dir = ROOT / "runs" / f"frontier_{args.arm}" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        i for i in range(args.paths)
        if not (args.resume and (out_dir / f"path_{i:04d}.json").exists())
    ]
    _log.info(
        "arm %s: %d strategies x %d paths (%d pending) on %d workers; "
        "block length %.2f, %d sessions",
        args.arm, len(entries), args.paths, len(pending), args.workers,
        block_length, len(sessions),
    )

    with RunManifest(cfg, script="run_frontier_stress.py") as run:
        run.note("arm", args.arm)
        run.note("n_strategies", len(entries))
        run.note("n_paths", args.paths)
        run.note("block_length", block_length)
        run.note("out_dir", str(out_dir.relative_to(ROOT)).replace("\\", "/"))
        completed = _dispatch(pending, paths, entries, args.workers, out_dir)
        run.note("paths_completed", len(completed))
    return 0 if len(completed) == len(pending) else 1


if __name__ == "__main__":
    sys.exit(main())
