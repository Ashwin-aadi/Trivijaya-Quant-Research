"""Split one frontier model's Markdown reply into individual candidate files, discarding nothing.

The frontier arms of the generator-validation study are collected by hand: the pre-registered prompt
is pasted into a chat interface and the whole reply is saved verbatim. This script is the only new
code the study needs. Everything downstream --- static audit, backtest, semantic audit, stress,
capacity --- is the frozen machinery P1 to P3 already built, and it consumes a directory of
``candidate_NNN.py`` files. So this script's single responsibility is to turn a reply into that
directory, and to do so without exercising any judgment about quality.

**Nothing is discarded, repaired or reformatted.** A block that does not parse is still written out
and still counted, because the pre-registration fixes the denominator as *strategies requested*
rather than *strategies that came back usable*. Silently dropping a malformed block would inflate
every rate the study reports, and the direction of that inflation is exactly the direction that
would flatter a frontier model. The raw reply is hashed before anything is read out of it, so the
artifact this ran against is pinned.

Usage:
    python scripts/extract_frontier_strategies.py \
        --raw benchmarks/generator_validation/raw/gpt_001.md \
        --arm gpt --model "GPT-5.2 Thinking"
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.log import configure_logging, get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]

#: A fenced block opened as ``python`` (or ``py``). Anything fenced differently is not a candidate
#: and is counted separately, because a model that answers in prose has failed the request in a way
#: worth distinguishing from one that answers with broken code.
_PYTHON_BLOCK = re.compile(r"^```(?:python|py)[ \t]*\r?\n(.*?)^```", re.DOTALL | re.MULTILINE)

#: Any fenced block at all, used only to report how many were not Python.
_ANY_BLOCK = re.compile(r"^```([A-Za-z0-9_+-]*)[ \t]*\r?\n.*?^```", re.DOTALL | re.MULTILINE)

#: The nearest preceding Markdown heading, recorded so a block can be tied to the theme it answers.
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Block:
    """One fenced Python block, with what could be established about it without changing it."""

    index: int
    filename: str
    heading: str
    n_chars: int
    parses: bool
    syntax_error: str | None
    strategy_classes: list[str]
    defines_generate: bool


def _heading_before(text: str, offset: int) -> str:
    """The last Markdown heading preceding ``offset``, or the empty string if there is none."""
    last = ""
    for match in _HEADING.finditer(text, 0, offset):
        last = match.group(1).strip()
    return last


def _strategy_classes(tree: ast.Module) -> list[str]:
    """Names of classes declaring a base called ``Strategy``.

    Matched on the attribute name alone, so both ``Strategy`` and ``strategy.Strategy`` count. The
    interface contract is what the study measures compliance with; how it was imported is not.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if name == "Strategy":
                found.append(node.name)
                break
    return found


def _describe(source: str, index: int, heading: str, filename: str) -> Block:
    """Everything establishable about one block by reading it. Never modifies it."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Block(index, filename, heading, len(source), False, f"{exc}", [], False)
    classes = _strategy_classes(tree)
    has_generate = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "generate"
        for n in ast.walk(tree)
    )
    return Block(index, filename, heading, len(source), True, None, classes, has_generate)


def extract(raw: str) -> list[tuple[str, str]]:
    """Every Python-fenced block, as ``(source, preceding heading)``, in document order."""
    return [
        (match.group(1), _heading_before(raw, match.start()))
        for match in _PYTHON_BLOCK.finditer(raw)
    ]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="the saved verbatim reply")
    parser.add_argument("--arm", required=True, help="short arm name, e.g. gpt")
    parser.add_argument("--model", required=True, help="product name exactly as the UI showed it")
    args = parser.parse_args()

    raw_bytes = args.raw.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    raw = raw_bytes.decode("utf-8")
    _log.info("raw reply %s, %d bytes, sha256 %s", args.raw.name, len(raw_bytes), digest[:16])

    out = ROOT / "runs" / f"frontier_{args.arm}" / "candidates"
    out.mkdir(parents=True, exist_ok=True)

    blocks = extract(raw)
    fence_kinds = [m.group(1) or "(none)" for m in _ANY_BLOCK.finditer(raw)]
    described: list[Block] = []
    for index, (source, heading) in enumerate(blocks):
        filename = f"candidate_{index:03d}.py"
        (out / filename).write_text(source, encoding="utf-8")
        described.append(_describe(source, index, heading, filename))

    record = {
        "arm": args.arm,
        "model_as_shown": args.model,
        "raw_file": str(args.raw.resolve()).replace(str(ROOT) + "\\", "").replace("\\", "/"),
        "raw_sha256": digest,
        "raw_chars": len(raw),
        "fenced_blocks_total": len(fence_kinds),
        "fence_kinds": sorted(set(fence_kinds)),
        "python_blocks": len(blocks),
        "blocks": [asdict(b) for b in described],
    }
    (out.parent / "extraction.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    parsed = sum(1 for b in described if b.parses)
    conforming = sum(1 for b in described if b.strategy_classes and b.defines_generate)
    _log.info("  fenced blocks: %d (%s)", len(fence_kinds), ", ".join(sorted(set(fence_kinds))))
    _log.info("  python blocks written: %d -> %s", len(blocks), out.relative_to(ROOT))
    _log.info("  parse as Python: %d of %d", parsed, len(blocks))
    _log.info("  declare a Strategy subclass with generate(): %d of %d", conforming, len(blocks))
    for b in described:
        if not b.parses:
            _log.warning("  %s does not parse, kept anyway: %s", b.filename, b.syntax_error)
        elif not b.strategy_classes:
            _log.warning("  %s parses but declares no Strategy subclass, kept anyway", b.filename)

    if len(blocks) != 5:
        _log.warning(
            "  %d python blocks, not the 5 requested. This is a datum, not an error to fix.",
            len(blocks),
        )

    with RunManifest(load_config(), "extract_frontier_strategies") as run:
        run.add_input(args.raw)
        for key, value in record.items():
            run.note(key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
