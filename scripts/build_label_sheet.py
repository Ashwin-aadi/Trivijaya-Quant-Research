"""Assemble the 50-item sheet a human reviewer labels by hand, to validate the semantic auditor.

Single responsibility: turn the fixture corpus into exactly fifty (rationale, code) pairs, shuffled
reproducibly, and write them in two forms — a markdown sheet for a person to read and a CSV for the
labels to be joined back on. Cohen's kappa between those hand labels and the auditor's labels is
computed elsewhere, by :mod:`src.eval.agreement`.

Usage (from the repository root)::

    python scripts/build_label_sheet.py

Three things about the construction matter more than the code.

*The sheet carries no answer key.* Neither output file records what any item is expected to be, and
neither says where an item came from beyond the source path the CSV needs for the join. A reviewer
who has been told the answer is no longer an independent rater, and a kappa computed against an
anchored reviewer measures nothing. The construction notes stay in this file; the deliberate mix
below is described only in aggregate, so reading this script does not hand over a per-item key
either.

*The reviewer and the model must see the same thing.* The ``code_excerpt`` column is the exact text
the semantic auditor should be given for that item. If the model reads the whole file and the
person reads a fragment, the two are answering different questions and their agreement is not
interpretable. Excerpting therefore happens once, here, and both raters consume it.

*The hard classes have to be present to be measured.* Thirty honest fixtures and three cheats, all
paired with their own real rationale, would leave the sheet almost entirely ``consistent``: kappa
would then be dominated by a single class and would say nothing about whether either rater can spot
a mismatch. Seventeen items therefore pair a real strategy's code with a substituted rationale,
drawn deliberately across the three defect families the taxonomy names. That is a limitation to
state plainly in the checkpoint report — the sheet's class balance is engineered, not natural, so
it measures whether the raters can tell these classes apart, not how often the classes occur.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.log import get_logger  # noqa: E402
from src.common.manifest import RunManifest  # noqa: E402

_log = get_logger("build_label_sheet")

CLEAN_DIR = Path("tests/fixtures/clean")
LEAKY_DIR = Path("tests/fixtures/leaky")
OUTPUT_DIR = Path("benchmarks/alphaaudit")
MARKDOWN_NAME = "label_sheet.md"
CSV_NAME = "label_sheet.csv"

#: Fixed by the charter's Phase 1.3 requirement: the PI hand-labels fifty examples.
SHEET_SIZE = 50

#: The shuffle seed. Global seed 42 per charter RULE 6, so the sheet regenerates byte-identically.
SEED = 42

#: A class body longer than this is cut to its docstring and ``generate`` method. Chosen so that a
#: reviewer working through fifty items sees a whole screen of code rather than a scroll, and set
#: after measuring the corpus: most fixtures fit under it untouched, so trimming is the exception
#: and is marked in the excerpt wherever it happens.
MAX_EXCERPT_LINES = 40


@dataclass(frozen=True)
class LabelItem:
    """One row of the sheet: what the reviewer reads, plus provenance for the join.

    ``origin`` is deliberately *not* written to either output. It is provenance for this script's
    own reporting and for the tests; in the sheet it would function as an answer key, since a
    reviewer who knows an item was constructed knows it is not ``consistent``.
    """

    item_id: str
    source_file: str
    rationale: str
    code_excerpt: str
    origin: str
    trimmed: bool


@dataclass(frozen=True)
class FixtureSource:
    """A strategy fixture reduced to the two things the audit task needs."""

    source_file: str
    rationale: str
    code_excerpt: str
    trimmed: bool


# Seventeen substituted rationales, each attached to the named fixture's real code. They are held
# in one flat sequence with no per-item annotation on purpose: an intent column here would be an
# answer key one copy-paste away from the sheet. In aggregate the seventeen were written to cover
# all three defect families the taxonomy names, so that none of them is starved in the sample.
_SUBSTITUTED_RATIONALES: tuple[tuple[str, str], ...] = (
    (
        "simple_momentum_63d",
        "Sharp moves over a handful of sessions are usually liquidity demand rather than news, "
        "and they unwind once that demand has been met. The rule therefore ranks the universe on "
        "the past week's return and holds the ten worst performers, waiting for the bounce.",
    ),
    (
        "high_volatility",
        "Calmer names have historically delivered better risk-adjusted returns than turbulent "
        "ones. The portfolio ranks the universe by realised volatility and holds the quietest "
        "ten, which makes it defensive by construction and should cause it to lag in a rally.",
    ),
    (
        "sma_crossover",
        "A name is held only while its price sits above its 200-day average, and every position "
        "carries an eight percent trailing stop that liquidates it into cash on breach. The long "
        "average defines the trend; the stop caps the damage when the trend ends.",
    ),
    (
        "breakout_20d",
        "Names sitting at the bottom of their recent range are where accumulated selling has "
        "already been absorbed, so the rule buys those closing nearest their twenty-session low "
        "and avoids anything making new highs.",
    ),
    (
        "rsi_oversold",
        "A high relative strength reading means buyers have controlled the tape for a sustained "
        "stretch, and that control tends to persist over the following weeks. The portfolio holds "
        "the names whose RSI has pushed above seventy.",
    ),
    (
        "equal_weight_universe",
        "The universe is screened on balance-sheet quality — return on equity above the median "
        "and net debt below one times operating profit — and only the names clearing both tests "
        "are held. Quality is the one characteristic that has not been arbitraged away.",
    ),
    (
        "inverse_volatility_weighted",
        "Each name receives a weight proportional to its conviction score, formed by combining "
        "three inputs: twelve-month momentum, earnings revision breadth, and analyst dispersion. "
        "Names below the conviction threshold are dropped outright rather than held small.",
    ),
    (
        "mean_reversion_5d",
        "The portfolio buys the largest recent decliners. What drives the return is the "
        "collective mood of participants, which reasserts itself over whatever horizon the "
        "conditions of the moment call for. Where the bounce is quick the position is rewarded "
        "quickly; where it is slow, the same force is simply taking longer to express itself, and "
        "a stretch without a payoff shows only that the horizon has not yet run its course.",
    ),
    (
        "trend_persistence",
        "Ranking on the share of advancing sessions selects the names carrying genuine conviction "
        "beneath the price, and conviction of that kind is recognised in the end. A period in "
        "which these names underperform is not evidence against the mechanism; it is a "
        "measurement taken before the recognition had arrived.",
    ),
    (
        "range_compression",
        "A narrowing daily range stores energy the market must eventually release, and holding "
        "the most compressed names captures that release. When it resolves upward the position "
        "profits directly. When it resolves downward, the compression had not finished building, "
        "and the name is simply selected again on better terms.",
    ),
    (
        "turnover_growth",
        "Attention arrives at a name before its price has finished adjusting, and the rupees "
        "changing hands are the visible trace of that attention. Where the price follows the "
        "turnover, the adjustment is under way; where it does not, the attention has not finished "
        "arriving and the adjustment is still ahead. The signal is early rather than wrong.",
    ),
    (
        "donchian_channel",
        "Where a name sits within its high-low band locates it in the market's natural cycle. The "
        "rule holds only the names near the top of the band, on the view that the cycle is "
        "advancing there. Where those names fall back instead, the cycle had already turned, "
        "which the same framework describes just as well.",
    ),
    (
        "momentum_skip_month",
        "Ranking on the trailing year while discarding the most recent month is a construction we "
        "arrived at internally and have not seen described anywhere. Dropping that month removes "
        "a short-horizon effect working against the longer one, and the spread that remains is "
        "the core of the edge here. It is not available in the published literature.",
    ),
    (
        "low_volatility",
        "Our own work on this universe turned up a relationship we did not expect and have found "
        "no prior account of: the calmest names in the index earn risk-adjusted returns at least "
        "as good as the turbulent ones. Standard risk-return reasoning says the opposite, which "
        "is presumably why nobody has been exploiting it.",
    ),
    (
        "bollinger_reversion",
        "A finding original to this framework: names that fall a long way below their recent "
        "average in a short space of time recover a meaningful part of that move over the "
        "following days. The effect is clearly measurable on this universe and, so far as we can "
        "establish, undocumented.",
    ),
    (
        "relative_strength_vs_universe",
        "We have isolated a proprietary relationship: names that have outrun the average of their "
        "own index over the past quarter go on outrunning it. This is not the market factor and "
        "it is not in the published cross-section. It emerged from our own screening work and it "
        "is the whole basis of the selection here.",
    ),
    (
        "price_vs_vwap",
        "This framework has surfaced an effect we believe to be new: names trading below the "
        "price at which the period's money actually transacted go on to outperform. Nothing in "
        "the literature we have reviewed anticipates it, and it appears to be a genuine and "
        "previously unclaimed source of return.",
    ),
)


# --- reading a fixture ---------------------------------------------------------


def _is_docstring(statement: ast.stmt) -> bool:
    """True if this statement is a bare string expression, i.e. a docstring."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _strategy_class(tree: ast.Module, path: Path) -> ast.ClassDef:
    """The single ``Strategy`` subclass a fixture module defines."""
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "Strategy" for base in node.bases)
    ]
    if len(classes) != 1:
        raise ValueError(f"{path} defines {len(classes)} Strategy subclasses, expected exactly 1")
    return classes[0]


def _rationale_assignment(node: ast.ClassDef, path: Path) -> ast.Assign:
    """The class-level ``rationale = ...`` statement, which every fixture is required to carry."""
    for statement in node.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "rationale"
            for target in statement.targets
        ):
            return statement
    raise ValueError(f"{path} has no class-level `rationale` assignment")


def _collapse_blanks(lines: list[str]) -> list[str]:
    """Trim leading and trailing blank lines and squeeze internal runs down to one.

    Removing the rationale assignment from the middle of a class leaves a gap; without this the
    excerpt would carry a visible hole exactly where the substituted items differ from the
    originals, which is the one place a reviewer must not be given a hint.
    """
    kept: list[str] = []
    for line in lines:
        if not line.strip() and (not kept or not kept[-1].strip()):
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _trimmed_class(lines: list[str], node: ast.ClassDef) -> str:
    """The class header, its docstring, and ``generate`` — with the omissions named in a comment.

    Naming what was left out is the point. An excerpt that silently drops a helper ``generate``
    depends on would have a reviewer judging code they cannot see, and neither rater should be
    guessing at the contents of a method nobody showed them.
    """
    first_body_line = node.body[0].lineno
    parts: list[str] = list(lines[node.lineno - 1 : first_body_line - 1])
    if _is_docstring(node.body[0]):
        doc = node.body[0]
        parts += lines[doc.lineno - 1 : (doc.end_lineno or doc.lineno)]
        parts.append("")
    generate = next(
        (s for s in node.body if isinstance(s, ast.FunctionDef) and s.name == "generate"), None
    )
    if generate is not None:
        parts += lines[generate.lineno - 1 : (generate.end_lineno or generate.lineno)]
    omitted = sorted(
        s.name for s in node.body if isinstance(s, ast.FunctionDef) and s.name != "generate"
    )
    note = "    # [excerpt: class docstring and generate() only"
    if omitted:
        note += "; " + ", ".join(f"{name}()" for name in omitted) + " not shown"
    parts += ["", note + "]"]
    return "\n".join(_collapse_blanks(parts))


def _code_excerpt(lines: list[str], node: ast.ClassDef, path: Path) -> tuple[str, bool]:
    """The strategy's code as both raters will see it, and whether it had to be cut.

    The module docstring, the imports and the ``rationale`` assignment are all dropped. The first
    two are not the strategy. The third has to go: leaving it in would print the fixture's own
    rationale directly underneath the one being audited, which gives away every substituted item
    and makes the whole exercise pointless.
    """
    drop = _rationale_assignment(node, path)
    span = (drop.lineno, drop.end_lineno or drop.lineno)
    kept = _collapse_blanks(
        [
            lines[number - 1]
            for number in range(node.lineno, (node.end_lineno or node.lineno) + 1)
            if not span[0] <= number <= span[1]
        ]
    )
    if len(kept) <= MAX_EXCERPT_LINES:
        return "\n".join(kept), False
    return _trimmed_class(lines, node), True


def load_fixture(path: Path, repo_root: Path) -> FixtureSource:
    """Read one fixture file into a rationale and a code excerpt."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    node = _strategy_class(tree, path)
    rationale = ast.literal_eval(_rationale_assignment(node, path).value)
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{path} has a rationale that is not a non-empty string")
    excerpt, trimmed = _code_excerpt(lines, node, path)
    return FixtureSource(
        source_file=path.relative_to(repo_root).as_posix(),
        rationale=" ".join(rationale.split()),
        code_excerpt=excerpt,
        trimmed=trimmed,
    )


# --- assembling the sheet ------------------------------------------------------


def _item_id(source_file: str, rationale: str) -> str:
    """A stable id derived from the pair itself, not from its position in the sheet.

    Position-based numbering would silently re-key every already-collected label the moment a
    fixture is added or the seed changes. Hashing the content means an item keeps its id across
    any reordering, so a partially filled sheet stays joinable.
    """
    digest = hashlib.sha256(f"{source_file}\0{rationale}".encode()).hexdigest()
    return f"itm-{digest[:8]}"


def _make_item(fixture: FixtureSource, rationale: str, origin: str) -> LabelItem:
    return LabelItem(
        item_id=_item_id(fixture.source_file, rationale),
        source_file=fixture.source_file,
        rationale=rationale,
        code_excerpt=fixture.code_excerpt,
        origin=origin,
        trimmed=fixture.trimmed,
    )


def build_items(repo_root: Path, seed: int = SEED) -> list[LabelItem]:
    """The full sheet: every fixture with its own rationale, plus the substituted pairs, shuffled.

    The shuffle is what stops the ordering itself from being a hint — unshuffled, the sheet would
    run thirty honest items, three cheats, then seventeen constructed ones, and a reviewer would
    read the pattern long before item fifty.
    """
    clean = {
        path.stem: load_fixture(path, repo_root)
        for path in sorted((repo_root / CLEAN_DIR).glob("*.py"))
        if not path.name.startswith("_")
    }
    leaky = [
        load_fixture(path, repo_root)
        for path in sorted((repo_root / LEAKY_DIR).glob("*.py"))
        if not path.name.startswith("_")
    ]

    items = [_make_item(fixture, fixture.rationale, "clean") for fixture in clean.values()]
    items += [_make_item(fixture, fixture.rationale, "leaky") for fixture in leaky]
    for stem, rationale in _SUBSTITUTED_RATIONALES:
        if stem not in clean:
            raise ValueError(f"substituted rationale names an unknown fixture: {stem}")
        items.append(_make_item(clean[stem], " ".join(rationale.split()), "constructed"))

    if len(items) != SHEET_SIZE:
        raise ValueError(f"expected {SHEET_SIZE} items, assembled {len(items)}")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("item ids collided; two items share a source file and a rationale")
    random.Random(seed).shuffle(items)
    return items


# --- writing the two outputs ---------------------------------------------------

_HEADER = """\
# Semantic audit — hand-labelling sheet

Fifty candidate strategies. Each item gives an author's stated economic rationale and the code that
implements it. Assign exactly one label per item, judging only whether the rationale is an honest
description of what the code does.

These same fifty items are put to the local model that forms the semantic audit layer. Cohen's
kappa between your labels and its labels is the measurement, so your labels must be formed
independently: work through the sheet in order and do not consult the model's output, the source
files, or the CSV's provenance columns until you have finished.

## The four labels

- **rationale_implementation_mismatch** — the code does not do what the rationale claims (a
  different window, a different direction, a rule that is described but never applied).
- **unfalsifiable_mechanism** — the rationale is phrased so that no outcome could contradict it; it
  would explain a profit and a loss equally well.
- **unacknowledged_known_anomaly** — a long-documented effect (momentum, low volatility, size,
  value, short-term reversal) is presented as novel, proprietary, or newly discovered. Implementing
  a known effect is not a defect; claiming to have found it is.
- **consistent** — none of the above. The rationale describes what the code does. It may still be
  wrong about the market; that is not the question here.

## Rules

1. Judge the code as written. Do not speculate about what the author meant.
2. If you are unsure, answer `consistent`.
3. Being simple, unoriginal, or unprofitable is not a defect.
4. If more than one label applies, use the first that applies in the order listed above.

## What you are shown

Each excerpt is the strategy class: its docstring and its methods, with imports and the module
docstring left out. The class's own `rationale` attribute is removed from every excerpt, so the
rationale printed above the code is the only one in play. Where a class was too long to print
whole, the excerpt is cut to the docstring and `generate`, and a comment inside the code block
names what was left out.

Write your answer on the `Your label:` line, then transcribe the fifty labels into the
`human_label` column of `label_sheet.csv`, matching on the item id.

---
"""


def render_markdown(items: list[LabelItem]) -> str:
    """The reviewer-facing sheet."""
    blocks = [_HEADER]
    for position, item in enumerate(items, start=1):
        blocks.append(
            f"\n### Item {position:02d} — `{item.item_id}`\n"
            f"\n**Rationale**\n"
            f"\n{item.rationale}\n"
            f"\n**Code**\n"
            f"\n```python\n{item.code_excerpt}\n```\n"
            f"\nYour label: ______________________________\n"
            f"\n---\n"
        )
    return "".join(blocks)


def write_csv(items: list[LabelItem], path: Path) -> None:
    """The machine-readable sheet, with ``human_label`` left empty for the reviewer."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "source_file", "rationale", "code_excerpt", "human_label"])
        for item in items:
            writer.writerow(
                [item.item_id, item.source_file, item.rationale, item.code_excerpt, ""]
            )


def main() -> int:
    cfg = load_config(_REPO_ROOT / "config" / "config.yaml")
    output_dir = _REPO_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with RunManifest(cfg, script="scripts/build_label_sheet.py") as run:
        items = build_items(_REPO_ROOT, seed=SEED)
        (output_dir / MARKDOWN_NAME).write_text(render_markdown(items), encoding="utf-8")
        write_csv(items, output_dir / CSV_NAME)

        counts = {origin: sum(1 for i in items if i.origin == origin) for origin in
                  ("clean", "leaky", "constructed")}
        trimmed = sum(1 for item in items if item.trimmed)
        run.note("sheet_items", len(items))
        run.note("sheet_origin_counts", counts)
        run.note("sheet_trimmed_excerpts", trimmed)
        run.note("sheet_seed", SEED)

    _log.info("wrote %d items to %s", len(items), output_dir / MARKDOWN_NAME)
    print(f"items:      {len(items)}")
    print(f"  clean fixtures, own rationale:       {counts['clean']}")
    print(f"  leaky fixtures, own rationale:       {counts['leaky']}")
    print(f"  substituted rationale on real code:  {counts['constructed']}")
    print(f"excerpts trimmed to docstring+generate: {trimmed}")
    print(f"markdown:   {output_dir / MARKDOWN_NAME}")
    print(f"csv:        {output_dir / CSV_NAME}")
    print("\nhuman_label is empty by design. Neither file records an expected label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
