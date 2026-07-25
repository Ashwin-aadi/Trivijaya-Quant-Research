"""Static detection of lookahead bias by reading a strategy's source, not its results.

**Why this layer must not reason from performance.** The obvious way to spot a cheating strategy
is that it makes too much money — but that only catches the cartoonish cases. A survivorship-biased
backtest in this repository produces a Sharpe of about 1.2: entirely plausible, the kind of number
a researcher would nod at and put into production. Anything detecting leakage by magnitude would
miss exactly the cases that matter. So this layer never looks at returns. It parses the code and
finds the cheat structurally, which is the only method that generalises to strategies whose results
look reasonable.

Detection is by AST inspection rather than regular expressions, because the patterns of interest
are structural. `shift(-1)` is a negative shift whether it is written as a literal, a unary minus,
or spread over three lines, and a regex tuned to the literal form silently misses the rest.

Each finding carries the class of defect, the line, the snippet, a severity, and an explanation in
plain English — the explanation is what a reviewer reads to decide whether they agree, so it states
what the code does and why that constitutes lookahead, never merely "suspicious pattern found".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LeakClass(StrEnum):
    """The kinds of lookahead this layer knows how to recognise."""

    FUTURE_INDEXING = "future_indexing"
    FULL_SAMPLE_FIT = "full_sample_fit"
    FULL_SAMPLE_STATISTIC = "full_sample_statistic"
    SURVIVORSHIP_SELECTION = "survivorship_selection"
    BOUNDARY_CROSSING_WINDOW = "boundary_crossing_window"
    TARGET_IN_FEATURES = "target_in_features"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Finding:
    """One detected defect, located precisely enough for a human to check it."""

    leak_class: LeakClass
    line_number: int
    code_snippet: str
    severity: Severity
    explanation: str


# Fitted transforms. Calling .fit() on any of these outside a training fold is the classic
# "scaler fitted before the split" leak.
_FITTED_TRANSFORMS = frozenset(
    {"StandardScaler", "MinMaxScaler", "RobustScaler", "PCA", "SimpleImputer", "KNNImputer",
     "QuantileTransformer", "PowerTransformer", "LabelEncoder", "OneHotEncoder"}
)

# Aggregations that collapse a whole series to one number. Harmless on a window; leakage when the
# series spans the entire sample, because the result encodes the future.
_AGGREGATIONS = frozenset({"mean", "std", "var", "median", "min", "max", "quantile", "sum"})

# Names that suggest the object being aggregated is the full panel rather than a visible window.
_FULL_SAMPLE_HINTS = ("panel", "full", "all_", "entire", "history_all", "dataset", "_data")

# Attribute or variable names implying end-of-period membership was consulted.
_SURVIVORSHIP_HINTS = (
    "final_", "last_rebalance", "survivor", "current_constituents", "latest_members",
    "end_of_period", "_today", "final_universe",
)

# Words that mark a variable as the thing being predicted. Seeing one among the features means the
# answer is in the inputs.
_TARGET_HINTS = ("target", "label", "y_true", "future_return", "next_return", "forward_return")


# Constructor parameters that hand a strategy bulk data rather than a setting. An honest strategy
# is configured with windows and thresholds; it receives its data through the point-in-time view at
# decision time. Taking the panel up front is how a strategy acquires the future.
_BULK_DATA_PARAMS = frozenset(
    {"panel", "prices", "data", "universe", "df", "frame", "returns", "history", "closes"}
)

# The single sanctioned data source inside a decision function.
_SANCTIONED_SOURCE = "view"


class _Visitor(ast.NodeVisitor):
    """Walks a module and records every leakage pattern it recognises."""

    def __init__(self, source_lines: list[str]) -> None:
        self.findings: list[Finding] = []
        self._lines = source_lines
        # Attributes assigned from a bulk-data constructor argument. Reading one of these inside a
        # decision function is the out-of-band access checked for below.
        self._stored_bulk_attributes: set[str] = set()
        self._inside_decision_function = False

    # --- helpers -----------------------------------------------------------
    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        if 1 <= line <= len(self._lines):
            return self._lines[line - 1].strip()
        return ""

    def _add(self, node: ast.AST, leak: LeakClass, severity: Severity, why: str) -> None:
        self.findings.append(
            Finding(
                leak_class=leak,
                line_number=getattr(node, "lineno", 0),
                code_snippet=self._snippet(node),
                severity=severity,
                explanation=why,
            )
        )

    @staticmethod
    def _is_negative_number(node: ast.AST) -> bool:
        """True for -1, -2, ... however they are spelled in the tree."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = node.operand
            return isinstance(operand, ast.Constant) and isinstance(operand.value, int | float)
        return isinstance(node, ast.Constant) and isinstance(node.value, int | float) \
            and node.value < 0

    @staticmethod
    def _mentions(name: str, hints: tuple[str, ...]) -> bool:
        lowered = name.lower()
        return any(hint in lowered for hint in hints)

    # --- visitors ----------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        """Inspect constructors for bulk data, and decision functions for out-of-band reads."""
        if node.name == "__init__":
            self._check_bulk_data_ingestion(node)
            self._record_stored_attributes(node)
            self.generic_visit(node)
            return

        # `generate` is the decision function: it is handed a point-in-time view and must use
        # nothing else. Anything it reads from instance state predates the view's truncation.
        if node.name == "generate":
            previous = self._inside_decision_function
            self._inside_decision_function = True
            self.generic_visit(node)
            self._inside_decision_function = previous
            return
        self.generic_visit(node)

    def _check_bulk_data_ingestion(self, node: ast.FunctionDef) -> None:
        """A constructor accepting the whole dataset rather than parameters."""
        for argument in node.args.args:
            if argument.arg == "self":
                continue
            if argument.arg.lower() in _BULK_DATA_PARAMS:
                self._add(
                    node, LeakClass.FULL_SAMPLE_FIT, Severity.HIGH,
                    f"the constructor accepts `{argument.arg}`, handing the strategy bulk data "
                    "before any decision date exists. An honest strategy is configured with "
                    "windows and thresholds and receives data through the point-in-time view at "
                    "decision time; anything derived here spans the whole sample, future included.",
                )

    def _record_stored_attributes(self, node: ast.FunctionDef) -> None:
        """Note which attributes hold constructor-supplied bulk data."""
        bulk = {a.arg.lower() for a in node.args.args if a.arg.lower() in _BULK_DATA_PARAMS}
        if not bulk:
            return
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    self._stored_bulk_attributes.add(target.attr)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self._check_out_of_band_read(node)
        if self._mentions(node.attr, _SURVIVORSHIP_HINTS):
            self._add(
                node, LeakClass.SURVIVORSHIP_SELECTION, Severity.HIGH,
                f"`{node.attr}` refers to membership as at the end of the period. Selecting the "
                "universe this way silently removes every company that later failed, so the "
                "backtest only ever trades survivors.",
            )
        self.generic_visit(node)

    def _check_out_of_band_read(self, node: ast.Attribute) -> None:
        """Reading stored bulk data from inside the decision function."""
        if not self._inside_decision_function:
            return
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            return
        if node.attr in self._stored_bulk_attributes:
            self._add(
                node, LeakClass.FUTURE_INDEXING, Severity.HIGH,
                f"`self.{node.attr}` is read while deciding, but it holds the dataset captured "
                f"at construction. The `{_SANCTIONED_SOURCE}` argument is truncated to before the "
                "decision moment; this attribute is not, so reading it reaches past that boundary "
                "and can return rows the strategy could not yet have seen.",
            )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API naming
        func = node.func
        if isinstance(func, ast.Attribute):
            self._check_shift(node, func)
            self._check_fit(node, func)
            self._check_full_sample_aggregation(node, func)
        self.generic_visit(node)

    def _check_shift(self, node: ast.Call, func: ast.Attribute) -> None:
        """A negative shift pulls a later row backwards onto an earlier one."""
        if func.attr not in ("shift", "diff"):
            return
        for argument in list(node.args) + [kw.value for kw in node.keywords]:
            if self._is_negative_number(argument):
                self._add(
                    node, LeakClass.FUTURE_INDEXING, Severity.HIGH,
                    f"`{func.attr}` is called with a negative period, which moves a future "
                    "observation onto the current row. Any feature built this way already "
                    "contains the outcome it is meant to predict.",
                )

    def _check_fit(self, node: ast.Call, func: ast.Attribute) -> None:
        """A transform fitted where the data is not restricted to a training fold."""
        if func.attr not in ("fit", "fit_transform"):
            return
        target = func.value
        name = target.id if isinstance(target, ast.Name) else \
            target.attr if isinstance(target, ast.Attribute) else ""
        constructed = isinstance(target, ast.Call) and isinstance(target.func, ast.Name) \
            and target.func.id in _FITTED_TRANSFORMS
        if constructed or self._mentions(name, ("scaler", "pca", "imputer", "encoder")):
            self._add(
                node, LeakClass.FULL_SAMPLE_FIT, Severity.HIGH,
                f"`{func.attr}` is applied to a transform outside a training fold. The fitted "
                "statistics then summarise data the strategy has not yet seen at decision time, "
                "so every later value is scaled using knowledge of the future.",
            )

    def _check_full_sample_aggregation(self, node: ast.Call, func: ast.Attribute) -> None:
        """An aggregate over something that looks like the whole panel, not a window."""
        if func.attr not in _AGGREGATIONS:
            return
        base = func.value
        # Only flag when the object aggregated is named like the full dataset. Aggregating a
        # rolling window is ordinary and must not be reported, or the layer flags everything.
        name = ""
        if isinstance(base, ast.Name):
            name = base.id
        elif isinstance(base, ast.Attribute):
            name = base.attr
        elif isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            name = base.value.id
        if name and self._mentions(name, _FULL_SAMPLE_HINTS):
            self._add(
                node, LeakClass.FULL_SAMPLE_STATISTIC, Severity.MEDIUM,
                f"`{func.attr}()` is computed over `{name}`, whose name indicates the complete "
                "sample rather than a trailing window. A statistic spanning the full period "
                "encodes where prices eventually went.",
            )

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if self._mentions(node.id, _SURVIVORSHIP_HINTS):
            self._add(
                node, LeakClass.SURVIVORSHIP_SELECTION, Severity.HIGH,
                f"`{node.id}` names an end-of-period constituent set. Applying it to earlier "
                "dates excludes companies that were tradable then but delisted later.",
            )
        elif self._mentions(node.id, _TARGET_HINTS):
            self._add(
                node, LeakClass.TARGET_IN_FEATURES, Severity.HIGH,
                f"`{node.id}` names the quantity being predicted. If it reaches the feature set, "
                "the model is being shown the answer.",
            )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        """Slices that reach forward, e.g. `series[t + 1:]` or a reversed slice."""
        sliced = node.slice
        if isinstance(sliced, ast.Slice):
            lower, step = sliced.lower, sliced.step
            if step is not None and self._is_negative_number(step):
                self._add(
                    node, LeakClass.FUTURE_INDEXING, Severity.MEDIUM,
                    "A reversed slice iterates the series backwards from the end, which makes it "
                    "easy to read observations that postdate the decision point.",
                )
            if isinstance(lower, ast.BinOp) and isinstance(lower.op, ast.Add):
                self._add(
                    node, LeakClass.FUTURE_INDEXING, Severity.HIGH,
                    "The slice starts at an offset *after* the current index, so it selects "
                    "observations that had not occurred at decision time.",
                )
        self.generic_visit(node)


def audit_source(source: str, filename: str = "<string>") -> list[Finding]:
    """Parse ``source`` and return every leakage finding, ordered by line.

    A file that does not parse yields a single high-severity finding rather than an exception:
    the caller is auditing untrusted, possibly generated code, and a syntax error is a result to
    be recorded — and counted as a trial — not a crash.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [
            Finding(
                leak_class=LeakClass.FUTURE_INDEXING,
                line_number=exc.lineno or 0,
                code_snippet=(exc.text or "").strip(),
                severity=Severity.HIGH,
                explanation=f"source does not parse and cannot be audited: {exc.msg}",
            )
        ]
    visitor = _Visitor(source.splitlines())
    visitor.visit(tree)
    return sorted(visitor.findings, key=lambda f: (f.line_number, f.leak_class.value))


def audit_file(path: Path) -> list[Finding]:
    """Audit one source file."""
    return audit_source(path.read_text(encoding="utf-8"), filename=str(path))


def is_rejected(findings: list[Finding]) -> bool:
    """Reject on any high-severity finding.

    Medium and low findings are reported but do not by themselves reject, because the patterns
    behind them are ambiguous enough that treating them as fatal would flag honest strategies.
    """
    return any(f.severity is Severity.HIGH for f in findings)
