"""Static detection of lookahead bias by reading a strategy's source, not its results.

**Why this layer must not reason from performance.** The obvious way to spot a cheating strategy
is that it makes too much money — but that only catches the cartoonish cases. A survivorship-biased
backtest in this repository produces a Sharpe of about 1.2: entirely plausible, the kind of number
a researcher would nod at and put into production. Anything detecting leakage by magnitude would
miss exactly the cases that matter. So this layer never looks at returns.

**Why it must not reason from vocabulary either.** An earlier version matched variable names —
`final_`, `target`, `panel` and so on. Measured against strategies whose cheats were phrased in
different words, recall was 0.37, and only one of nine reworded cheats was caught. It also flagged
honest code: a strategy naming a local `target_weight` was rejected, and one naming the most recent
visible session `_today` was rejected on the very line implementing point-in-time stamping
correctly. A detector defeated by renaming a variable is not a detector.

**The principle this version uses instead: where did the data come from?**

The engine hands a strategy exactly one sanctioned source — the ``view`` argument of ``generate``,
already truncated to before the decision moment. Every leak in this domain is a read that traces
back to something else: a frame captured in the constructor, a module global, a value derived from
the whole sample. So the analysis tracks *provenance* rather than spelling. A constructor argument
is bulk data because it gets indexed and filtered, not because it is called ``panel``. Renaming
cannot defeat that, because no name is consulted.

Each finding carries the class of defect, the line, the snippet, a severity, and an explanation in
plain English — the explanation is what a reviewer reads to decide whether they agree, so it states
what the code does and why that constitutes lookahead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LeakClass(StrEnum):
    """The kinds of lookahead this layer recognises.

    ``POINT_IN_TIME_BYPASS`` names the defect of obtaining data outside the sanctioned channel —
    a frame captured in the constructor, a module global, an attribute read while deciding. It is
    reported separately from what the strategy then *does* with that data, because those are
    different defects and a corpus-level breakdown that merges them is misleading. A strategy that
    captures the panel and takes a percentile of it is committing both, and both are reported.
    """

    FUTURE_INDEXING = "future_indexing"
    FULL_SAMPLE_FIT = "full_sample_fit"
    FULL_SAMPLE_STATISTIC = "full_sample_statistic"
    SURVIVORSHIP_SELECTION = "survivorship_selection"
    BOUNDARY_CROSSING_WINDOW = "boundary_crossing_window"
    TARGET_IN_FEATURES = "target_in_features"
    SNOOPED_PARAMETER = "snooped_parameter"
    POINT_IN_TIME_BYPASS = "point_in_time_bypass"


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


# The function the engine calls to obtain a decision, and the argument through which it supplies
# the only data a strategy is entitled to see.
DECISION_FUNCTION = "generate"
SANCTIONED_SOURCE = "view"

# Methods that only make sense on a collection. Seeing any of them applied to a value is how this
# module decides that value holds data, rather than consulting what it is named.
DATA_METHODS = frozenset(
    {"filter", "select", "group_by", "groupby", "agg", "join", "with_columns", "pivot",
     "sort", "sort_values", "head", "tail", "unique", "drop_nulls", "dropna", "iter_rows",
     "to_list", "to_numpy", "to_pandas", "explode", "melt", "slice", "gather", "get_column",
     "mean", "std", "var", "median", "min", "max", "quantile", "sum", "count", "first", "last"}
)

# Aggregations that collapse a series to a single number.
AGGREGATIONS = frozenset(
    {"mean", "std", "var", "median", "min", "max", "quantile", "sum", "first", "last"}
)

# Transforms fitted to data. Fitting one outside a training fold is the classic scaler leak.
# These are the scikit-learn spellings; a method that fits under any other name is recognised from
# its structure instead, by `_collect_fitting_methods`.
FIT_METHODS = frozenset({"fit", "fit_transform", "partial_fit"})

# Window operations. Applied to a series spanning a split, these mix the two sides together.
WINDOW_METHODS = frozenset(
    {"rolling", "rolling_mean", "rolling_std", "rolling_sum", "rolling_min", "rolling_max",
     "rolling_median", "rolling_apply", "ewm", "ewm_mean", "resample", "upsample",
     "group_by_dynamic"}
)

# Operations that read a later value into an earlier row by construction. These leak regardless of
# where the data came from, so they are flagged on sight.
BACKWARD_FILL_METHODS = frozenset(
    {"interpolate", "bfill", "backfill", "backward_fill", "fill_null", "fillna", "pad_backward"}
)

# Keywords whose presence turns an otherwise-neutral fill into a backward one.
_BACKWARD_KEYWORDS = ("backward", "bfill", "backfill")


@dataclass
class _FunctionScope:
    """Names known to hold data inside one function, and how they got there."""

    tainted: set[str]          # traced back to something other than the sanctioned view
    from_future: set[str]      # derived from an explicitly forward-looking operation
    extrema: set[str]          # derived from a time-collapsing max/last, used for survivorship


class _Visitor(ast.NodeVisitor):
    """Walks a module and records every leakage pattern it recognises."""

    def __init__(self, source_lines: list[str]) -> None:
        self.findings: list[Finding] = []
        self._lines = source_lines
        # Attributes that were shown to hold bulk data, established while reading __init__.
        self._data_attributes: set[str] = set()
        self._module_data_names: set[str] = set()
        self._candidate_collections: set[str] = set()
        self._data_consumers: dict[str, set[int]] = {}
        self._fit_methods: set[str] = set()
        self._scope: _FunctionScope | None = None
        self._in_decision = False
        # How many enclosing calls have a tainted receiver. Chained frame work nests the interesting
        # operation inside the call that establishes provenance — `panel.sort(...).with_columns(
        # pl.col("x").rolling_mean(...))` — where the rolling window's own receiver is a bare column
        # expression carrying no provenance at all. Reading only the immediate receiver misses it.
        self._tainted_depth = 0

    # --- helpers -----------------------------------------------------------
    def _snippet(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        return self._lines[line - 1].strip() if 1 <= line <= len(self._lines) else ""

    def _add(self, node: ast.AST, leak: LeakClass, severity: Severity, why: str) -> None:
        self.findings.append(
            Finding(leak, getattr(node, "lineno", 0), self._snippet(node), severity, why)
        )

    @staticmethod
    def _is_negative_number(node: ast.AST) -> bool:
        """True for -1, -2, ... however they are spelled in the tree."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return isinstance(node.operand, ast.Constant) and isinstance(
                node.operand.value, int | float
            )
        return (isinstance(node, ast.Constant) and isinstance(node.value, int | float)
                and node.value < 0)

    @staticmethod
    def _root_name(node: ast.AST) -> str | None:
        """The base identifier of an expression like ``a.b[c].d()`` — here, ``a``."""
        current: ast.AST = node
        while True:
            if isinstance(current, ast.Name):
                return current.id
            if isinstance(current, ast.Attribute | ast.Subscript):
                current = current.value
            elif isinstance(current, ast.Call):
                current = current.func
            else:
                return None

    def _root_attribute(self, node: ast.AST) -> str | None:
        """For ``self.x...``, the attribute name ``x``; otherwise None."""
        current: ast.AST = node
        while isinstance(current, ast.Call | ast.Subscript):
            current = current.func if isinstance(current, ast.Call) else current.value
        while isinstance(current, ast.Attribute):
            if isinstance(current.value, ast.Name) and current.value.id == "self":
                return current.attr
            current = current.value
        return None

    def _is_tainted(self, node: ast.AST) -> bool:
        """True if this expression reads data from anywhere but the sanctioned view."""
        attribute = self._root_attribute(node)
        if attribute is not None and attribute in self._data_attributes:
            return True
        root = self._root_name(node)
        if root is None:
            return False
        if root in self._module_data_names:
            return True
        return self._scope is not None and root in self._scope.tainted

    def _reads_tainted(self, node: ast.AST) -> bool:
        """As ``_is_tainted``, but also true anywhere inside a call on a tainted receiver."""
        return self._tainted_depth > 0 or self._is_tainted(node)

    def _mentions_tainted(self, node: ast.AST) -> bool:
        """True if any part of this expression reads tainted data.

        Used for propagation rather than detection. A helper call hides its provenance behind the
        function name — ``_trailing_returns(panel, lookback)`` has root ``_trailing_returns`` — so
        the receiver alone says nothing and the arguments have to be inspected.
        """
        return any(
            isinstance(child, ast.Name | ast.Attribute | ast.Subscript) and self._is_tainted(child)
            for child in ast.walk(node)
        )

    # --- constructor analysis ---------------------------------------------
    def _param_is_used_as_data(self, name: str, body: list[ast.stmt]) -> bool:
        """Decide by usage, not by name, whether a parameter carries bulk data.

        A configuration value is compared, stored, or used in arithmetic. A frame is indexed with
        a column name, filtered, aggregated, or iterated. That difference is visible in the tree
        and survives any renaming, which the previous word-list approach did not.
        """
        # Track what the parameter flows into, not just what is done to it directly. A constructor
        # commonly hands the frame to a helper — `returns = _prepare(panel, window)` — and then
        # aggregates the result. The data operation is one step removed from the parameter, so
        # looking only at `panel.<method>()` sees nothing.
        derived = {name}
        for node in body:
            for child in ast.walk(node):
                if not isinstance(child, ast.Assign):
                    continue
                mentions = {n.id for n in ast.walk(child.value) if isinstance(n, ast.Name)}
                if mentions & derived:
                    derived.update(t.id for t in child.targets if isinstance(t, ast.Name))

        for node in body:
            for child in ast.walk(node):
                if isinstance(child, ast.Subscript):
                    if (self._root_name(child.value) in derived
                            and isinstance(child.slice, ast.Constant)
                            and isinstance(child.slice.value, str)):
                        return True
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if (child.func.attr in DATA_METHODS
                            and self._root_name(child.func.value) in derived):
                        return True
                elif isinstance(child, ast.For) and self._root_name(child.iter) in derived:
                    return True
        return False

    def _record_data_attributes(self, name: str, body: list[ast.stmt]) -> None:
        """Note which ``self.x`` attributes end up holding something derived from ``name``."""
        for node in body:
            for child in ast.walk(node):
                if not isinstance(child, ast.Assign):
                    continue
                if self._root_name(child.value) != name:
                    continue
                for target in child.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"):
                        self._data_attributes.add(target.attr)

    def _param_reaches_a_data_attribute(self, name: str, body: list[ast.stmt]) -> bool:
        """True if this parameter is stored into an attribute the class treats as data."""
        for node in body:
            for child in ast.walk(node):
                if not isinstance(child, ast.Assign) or self._root_name(child.value) != name:
                    continue
                for target in child.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and target.attr in self._data_attributes):
                        return True
        return False

    def _param_flows_into_a_data_consumer(self, name: str, body: list[ast.stmt]) -> bool:
        """True if the parameter is handed to a function that treats that argument as data."""
        for node in body:
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                called = child.func.attr if isinstance(child.func, ast.Attribute) else (
                    child.func.id if isinstance(child.func, ast.Name) else None)
                if called is None or called not in self._data_consumers:
                    continue
                for position, argument in enumerate(child.args):
                    if (isinstance(argument, ast.Name) and argument.id == name
                            and position in self._data_consumers[called]):
                        return True
        return False

    def _visit_constructor(self, node: ast.FunctionDef) -> set[str]:
        """Report every parameter carrying bulk data, and return their names as tainted sources."""
        carriers: set[str] = set()
        for argument in node.args.args:
            if argument.arg == "self":
                continue
            # Two ways to establish that a parameter carries data: it is used as data right here,
            # or it is stored into an attribute the class uses as data elsewhere. The second case
            # is the common one — a constructor that only assigns gives nothing away by itself.
            if not (self._param_is_used_as_data(argument.arg, node.body)
                    or self._param_reaches_a_data_attribute(argument.arg, node.body)
                    or self._param_flows_into_a_data_consumer(argument.arg, node.body)):
                continue
            self._add(
                node, LeakClass.POINT_IN_TIME_BYPASS, Severity.HIGH,
                f"the constructor parameter `{argument.arg}` is indexed, filtered or aggregated "
                "inside __init__, so it carries a dataset rather than a setting. A strategy "
                "receives its data through the point-in-time view at decision time; anything "
                "derived here spans the whole sample, future included.",
            )
            self._record_data_attributes(argument.arg, node.body)
            carriers.add(argument.arg)
        return carriers

    # --- function dispatch -------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        if node.name == "__init__":
            carriers = self._visit_constructor(node)
            # The constructor is where tainted data enters, so it is the one function that must be
            # walked with provenance already established. Leaving it unscoped — as an earlier
            # version did — meant the percentile taken over the whole panel, the extremum feeding a
            # membership filter and the centred rolling window were all invisible, and the only
            # thing reported was that a parameter carried data. Every such case then bore the
            # constructor rule's label whatever it was actually doing.
            previous_scope, previous_flag = self._scope, self._in_decision
            self._scope = _FunctionScope(set(carriers), set(), set())
            self._in_decision = False
            # A parameter sweep is usually run once at construction, not per decision, so the
            # constructor has to be searched for it too.
            self._check_parameter_search(node)
            self.generic_visit(node)
            self._scope, self._in_decision = previous_scope, previous_flag
            return

        previous_scope, previous_flag = self._scope, self._in_decision
        self._scope = _FunctionScope(set(), set(), set())
        self._in_decision = node.name == DECISION_FUNCTION
        self._check_parameter_search(node)
        self.generic_visit(node)
        self._scope, self._in_decision = previous_scope, previous_flag

    def _check_parameter_search(self, node: ast.FunctionDef) -> None:
        """A loop that scores candidate parameter values against data and keeps the best.

        Selecting a parameter by evaluating it on the sample being tested is snooping, and unlike
        a bare hard-coded constant it leaves a visible structure: iteration over candidates, a
        data operation inside, and a running best updated under a comparison.
        """
        # A sweep is just as often written as a comprehension feeding max(), so both shapes are
        # examined. What identifies it is not the loop construct but the combination: iterate
        # candidates, score each against data, keep the winner.
        comprehensions = [n for n in ast.walk(node)
                          if isinstance(n, ast.DictComp | ast.ListComp | ast.SetComp
                                        | ast.GeneratorExp)]
        for comp in comprehensions:
            # The decisive question is *what* is being iterated. An honest strategy loops over
            # symbols — a collection derived from the data. A parameter sweep loops over candidate
            # settings, which are numeric literals written into the source. Only the second is
            # snooping, and confusing them would reject ordinary cross-sectional code.
            if not self._iterates_candidate_values(comp.generators[0].iter):
                continue
            scores_data = any(
                isinstance(n, ast.Call) for n in ast.walk(comp)
            )
            if scores_data and self._selects_a_winner(node):
                self._add(
                    comp, LeakClass.SNOOPED_PARAMETER, Severity.HIGH,
                    "candidate parameter values are scored against the data and the best is "
                    "kept. The parameter is chosen with knowledge of the sample it is later "
                    "evaluated on, so the reported performance includes the benefit of that "
                    "search.",
                )
                return

        for loop in [n for n in ast.walk(node) if isinstance(n, ast.For)]:
            if not self._iterates_candidate_values(loop.iter):
                continue
            body = list(ast.walk(ast.Module(body=loop.body, type_ignores=[])))
            touches_data = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in DATA_METHODS | AGGREGATIONS
                for n in body
            )
            keeps_best = any(isinstance(n, ast.Compare) for n in body) and any(
                isinstance(n, ast.Assign) for n in body
            )
            if touches_data and keeps_best:
                self._add(
                    loop, LeakClass.SNOOPED_PARAMETER, Severity.HIGH,
                    "this loop scores candidate parameter values against the data and keeps the "
                    "best one. The parameter is therefore chosen with knowledge of the sample it "
                    "is later evaluated on, so the reported performance includes the benefit of "
                    "that search.",
                )

    def _iterates_candidate_values(self, node: ast.AST) -> bool:
        """True if this iterable is a fixed set of parameter values rather than data.

        Numeric literals written into the source are settings being swept. A name bound at module
        level to such a literal counts too, since moving the list to a constant does not change
        what the loop is doing.
        """
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "range"):
            return True
        if isinstance(node, ast.List | ast.Tuple):
            return all(isinstance(e, ast.Constant) and isinstance(e.value, int | float)
                       for e in node.elts) and bool(node.elts)
        if isinstance(node, ast.Name):
            return node.id in self._candidate_collections
        return False

    @staticmethod
    def _selects_a_winner(node: ast.FunctionDef) -> bool:
        """True if the function picks an extremum out of a collection of scores."""
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in ("max", "min", "argmax", "argmin", "sorted")
            for n in ast.walk(node)
        )

    # --- assignments propagate provenance ----------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if self._scope is not None:
            self._propagate(node)
        self.generic_visit(node)

    def _propagate(self, node: ast.Assign) -> None:
        assert self._scope is not None
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            return
        value = node.value
        if self._mentions_tainted(value):
            self._scope.tainted.update(targets)
        if self._derives_from_future(value):
            self._scope.from_future.update(targets)
        if self._is_time_extremum(value):
            self._scope.extrema.update(targets)

    def _derives_from_future(self, node: ast.AST) -> bool:
        """True if the expression reads forward in time, or was built from something that did."""
        assert self._scope is not None
        for child in ast.walk(node):
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr in ("shift", "diff")):
                arguments = list(child.args) + [kw.value for kw in child.keywords]
                if any(self._is_negative_number(a) for a in arguments):
                    return True
            if isinstance(child, ast.Name) and child.id in self._scope.from_future:
                return True
        return False

    def _is_time_extremum(self, node: ast.AST) -> bool:
        """True for ``something.max()`` / ``.last()`` / ``[-1]`` — a collapse of the time axis."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return node.func.attr in ("max", "last")
        return isinstance(node, ast.Subscript) and self._is_negative_number(node.slice)

    # --- reads -------------------------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if (self._in_decision and isinstance(node.value, ast.Name)
                and node.value.id == "self" and node.attr in self._data_attributes):
            self._add(
                node, LeakClass.POINT_IN_TIME_BYPASS, Severity.HIGH,
                f"`self.{node.attr}` is read while deciding, but it holds the dataset captured at "
                f"construction. The `{SANCTIONED_SOURCE}` argument is truncated to before the "
                "decision moment; this attribute is not, so reading it reaches past that boundary.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute):
            self._check_shift(node, node.func)
            self._check_fit(node, node.func)
            self._check_backward_fill(node, node.func)
            self._check_window(node, node.func)
            self._check_tainted_aggregation(node, node.func)
            self._check_survivorship_filter(node, node.func)
        # Everything nested inside a call on a tainted receiver inherits that provenance, so the
        # window or aggregation buried in a chained frame expression is judged against the frame
        # it is really operating on rather than against the bare column expression it hangs off.
        inherits = isinstance(node.func, ast.Attribute) and self._is_tainted(node.func.value)
        if inherits:
            self._tainted_depth += 1
        self.generic_visit(node)
        if inherits:
            self._tainted_depth -= 1

    def _check_shift(self, node: ast.Call, func: ast.Attribute) -> None:
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
        """A transform fitted to data that is not restricted to a training fold."""
        if func.attr not in FIT_METHODS and func.attr not in self._fit_methods:
            return
        fitted_on_tainted = any(self._reads_tainted(a) for a in node.args)
        constructed_here = isinstance(func.value, ast.Call)
        if fitted_on_tainted or constructed_here or self._reads_tainted(func.value):
            self._add(
                node, LeakClass.FULL_SAMPLE_FIT, Severity.HIGH,
                f"`{func.attr}` fits a transform on data that is not confined to a training fold. "
                "The fitted statistics summarise observations the strategy has not seen at "
                "decision time, so every later value is scaled using knowledge of the future.",
            )

    def _check_backward_fill(self, node: ast.Call, func: ast.Attribute) -> None:
        """Filling or interpolating backwards pulls later values into earlier rows."""
        if func.attr not in BACKWARD_FILL_METHODS:
            return
        keyword_text = " ".join(
            str(kw.value.value) for kw in node.keywords
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
        ).lower()
        explicitly_backward = any(word in keyword_text for word in _BACKWARD_KEYWORDS)
        inherently_backward = func.attr in ("interpolate", "bfill", "backfill", "backward_fill")
        if explicitly_backward or inherently_backward:
            self._add(
                node, LeakClass.BOUNDARY_CROSSING_WINDOW, Severity.HIGH,
                f"`{func.attr}` fills earlier rows from later ones. Every value it writes is "
                "information that did not exist at the timestamp it is written to, so any window "
                "spanning the gap now straddles the train/test boundary.",
            )

    def _check_window(self, node: ast.Call, func: ast.Attribute) -> None:
        """A rolling or resampling window computed over a source wider than the visible history."""
        if func.attr not in WINDOW_METHODS:
            return
        # A centred window averages rows on both sides of the one it labels, so it reads forward by
        # construction whatever it is applied to. That is a property of the window itself and holds
        # even where provenance is clean, so it is judged independently of the source.
        centred = any(
            kw.arg in ("center", "centered") and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if centred:
            self._add(
                node, LeakClass.BOUNDARY_CROSSING_WINDOW, Severity.HIGH,
                f"`{func.attr}` is centred, so each output value averages the sessions after the "
                "row it is attached to as well as those before it. Every point in the smoothed "
                "series therefore contains information that did not exist at its own timestamp, "
                "and points near a split are built from rows on both sides of it.",
            )
        elif self._reads_tainted(func.value):
            self._add(
                node, LeakClass.BOUNDARY_CROSSING_WINDOW, Severity.HIGH,
                f"`{func.attr}` is applied to data that did not come from the point-in-time view, "
                "so the window spans the whole sample. Statistics computed before a split mix the "
                "training and test sides together.",
            )

    def _check_tainted_aggregation(self, node: ast.Call, func: ast.Attribute) -> None:
        """An aggregation over a source that reaches beyond the decision moment."""
        if func.attr not in AGGREGATIONS or not self._reads_tainted(func.value):
            return
        self._add(
            node, LeakClass.FULL_SAMPLE_STATISTIC, Severity.HIGH,
            f"`{func.attr}()` is computed over data that did not come from the point-in-time "
            "view. A statistic spanning the full period encodes where prices eventually went.",
        )

    def _check_survivorship_filter(self, node: ast.Call, func: ast.Attribute) -> None:
        """Filtering a membership table against its own latest date, then trading earlier dates.

        This is the structure of the survivorship cheat: collapse the time axis to its final value
        and apply the resulting constituent set backwards. It is detected from the shape of the
        filter rather than from what anything is called.
        """
        if func.attr not in ("filter", "loc", "query") or self._scope is None:
            return
        for argument in node.args:
            for child in ast.walk(argument):
                if isinstance(child, ast.Name) and child.id in self._scope.extrema:
                    self._add(
                        node, LeakClass.SURVIVORSHIP_SELECTION, Severity.HIGH,
                        f"the membership set is filtered against `{child.id}`, which holds the "
                        "latest value of a time column. Selecting the final snapshot and applying "
                        "it to earlier dates removes every constituent that was tradable then but "
                        "dropped out later, so only survivors are ever held.",
                    )
                    return

    # --- slices ------------------------------------------------------------
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if isinstance(node.slice, ast.Slice):
            lower, step = node.slice.lower, node.slice.step
            if step is not None and self._is_negative_number(step):
                self._add(
                    node, LeakClass.FUTURE_INDEXING, Severity.MEDIUM,
                    "a reversed slice iterates the series backwards from the end, which makes it "
                    "easy to read observations that postdate the decision point.",
                )
            if isinstance(lower, ast.BinOp) and isinstance(lower.op, ast.Add):
                self._add(
                    node, LeakClass.FUTURE_INDEXING, Severity.HIGH,
                    "the slice starts at an offset after the current index, so it selects "
                    "observations that had not occurred at decision time.",
                )
        self.generic_visit(node)

    # --- returned values ---------------------------------------------------
    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        """A value derived from a forward-looking operation reaching the output."""
        if self._in_decision and node.value is not None and self._scope is not None:
            for child in ast.walk(node.value):
                if isinstance(child, ast.Name) and child.id in self._scope.from_future:
                    self._add(
                        node, LeakClass.TARGET_IN_FEATURES, Severity.HIGH,
                        f"`{child.id}` was built from a forward-looking operation and is used in "
                        "the returned portfolio. The quantity being predicted is among the inputs, "
                        "so the strategy is shown the answer.",
                    )
                    break
        self.generic_visit(node)


def _collect_data_attributes(tree: ast.Module) -> set[str]:
    """Attributes that behave like data, judged across the whole module.

    A single pass over ``__init__`` is not enough. The commonest bypass stores the frame and does
    nothing else with it — ``self._panel = panel`` — so the constructor body offers no evidence at
    all, and the proof of what the attribute holds appears later, where ``generate`` filters it.

    So the question is asked of the class as a whole: is this attribute ever indexed with a column
    name, filtered, aggregated, or iterated? If so it holds a dataset, whatever it is called and
    whichever method reveals it.
    """
    attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            base = node.value
            if (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                    and base.value.id == "self"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                attributes.add(base.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in DATA_METHODS:
                continue
            base = node.func.value
            while isinstance(base, ast.Call | ast.Subscript):
                base = base.func if isinstance(base, ast.Call) else base.value
            if (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name)
                    and base.value.id == "self"):
                attributes.add(base.attr)
        elif isinstance(node, ast.For):
            target = node.iter
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                attributes.add(target.attr)
    return attributes


def _collect_data_consuming_functions(tree: ast.Module) -> dict[str, set[int]]:
    """For each function, which of its parameter positions are treated as data inside it.

    Needed because a constructor rarely touches the frame itself. The common shape is
    ``self._scaler = self._fit_over_everything(panel)`` — the parameter is handed straight to a
    helper and every data operation happens one call away. Looking only at the constructor body
    sees a bare argument pass and concludes nothing, which is how a real cheat slipped through.

    Positions are recorded both with and without a leading ``self``, since the same name may be a
    method or a free function and the call site does not say which.
    """
    consuming: dict[str, set[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        parameters = [a.arg for a in node.args.args]
        positions: set[int] = set()
        for index, parameter in enumerate(parameters):
            if parameter == "self":
                continue
            if _param_used_as_data_in(parameter, node.body):
                # Offset by one when the first parameter is self, so the index matches the
                # arguments actually written at the call site.
                positions.add(index - 1 if parameters and parameters[0] == "self" else index)
        if positions:
            consuming[node.name] = positions
    return consuming


def _param_used_as_data_in(name: str, body: list[ast.stmt]) -> bool:
    """Whether ``name`` is indexed, filtered, aggregated or iterated anywhere in ``body``."""
    derived = {name}
    for node in body:
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                mentions = {n.id for n in ast.walk(child.value) if isinstance(n, ast.Name)}
                if mentions & derived:
                    derived.update(t.id for t in child.targets if isinstance(t, ast.Name))
    for node in body:
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                base = child.value
                root = base.id if isinstance(base, ast.Name) else None
                if root in derived and isinstance(child.slice, ast.Constant) \
                        and isinstance(child.slice.value, str):
                    return True
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                base = child.func.value
                while isinstance(base, ast.Call | ast.Subscript):
                    base = base.func if isinstance(base, ast.Call) else base.value
                if isinstance(base, ast.Name) and base.id in derived \
                        and child.func.attr in DATA_METHODS:
                    return True
            elif isinstance(child, ast.For) and isinstance(child.iter, ast.Name) \
                    and child.iter.id in derived:
                return True
    return False


def _collect_fitting_methods(tree: ast.Module) -> set[str]:
    """Methods that fit a transform, recognised by what they do rather than what they are called.

    ``FIT_METHODS`` covers the scikit-learn spellings and nothing else, which makes it the same kind
    of word list this module removed everywhere else: a strategy whose normaliser exposes
    ``calibrate`` rather than ``fit`` walks straight past it.

    Fitting has a structure independent of naming. The method takes an argument it treats as data,
    derives summary statistics from it, and retains them on the instance for later use. Anything
    doing all three is a fit, and applying it outside a training fold leaks whatever period the
    argument spanned.
    """
    fitting: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "__init__":
            continue
        takes_data = any(
            argument.arg != "self" and _param_used_as_data_in(argument.arg, node.body)
            for argument in node.args.args
        )
        if not takes_data:
            continue
        summarises = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in AGGREGATIONS
            for n in ast.walk(node)
        )
        retains = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self" for t in n.targets)
            for n in ast.walk(node)
        )
        if summarises and retains:
            fitting.add(node.name)
    return fitting


def _collect_candidate_collections(tree: ast.Module) -> set[str]:
    """Module-level names bound to a list or tuple of numeric literals.

    A parameter sweep is often written against a named constant rather than an inline list. Moving
    the candidates into a constant does not change what the loop does, so the name is resolved here
    and treated the same way.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List | ast.Tuple):
            continue
        elements = node.value.elts
        if elements and all(isinstance(e, ast.Constant) and isinstance(e.value, int | float)
                            for e in elements):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _collect_module_data_names(tree: ast.Module) -> set[str]:
    """Module-level names that hold something behaving like a dataset.

    A global frame is a way of smuggling the full sample into a decision function without passing
    it through the constructor, so provenance tracking has to know about it.
    """
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        looks_like_data = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr in DATA_METHODS | {"read_parquet", "read_csv", "DataFrame"}
        )
        if looks_like_data:
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


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
            Finding(LeakClass.FUTURE_INDEXING, exc.lineno or 0, (exc.text or "").strip(),
                    Severity.HIGH, f"source does not parse and cannot be audited: {exc.msg}")
        ]
    visitor = _Visitor(source.splitlines())
    visitor._data_attributes = _collect_data_attributes(tree)
    visitor._candidate_collections = _collect_candidate_collections(tree)
    visitor._data_consumers = _collect_data_consuming_functions(tree)
    visitor._fit_methods = _collect_fitting_methods(tree)
    visitor._module_data_names = _collect_module_data_names(tree)
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
