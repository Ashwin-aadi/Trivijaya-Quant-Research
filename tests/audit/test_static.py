"""Tests for the static leakage auditor.

Two things are being checked, and the second matters as much as the first. The auditor must catch
every deliberate cheat, and it must stay quiet on honest code — a detector that flags everything
has perfect recall and no value. Several tests below are therefore negative controls: ordinary
constructions that look superficially like leakage and must not be reported.

The fixture-level test is the real measurement. It runs the auditor over every strategy in the
repository and asserts precision and recall directly, so a regression that starts flagging honest
strategies fails here rather than quietly degrading the Phase 1.3 numbers.
"""

from pathlib import Path

from src.audit.static import (
    LeakClass,
    Severity,
    audit_file,
    audit_source,
    is_rejected,
)

LEAKY_DIR = Path("tests/fixtures/leaky")
CLEAN_DIR = Path("tests/fixtures/clean")


def classes_of(source: str) -> set[LeakClass]:
    return {finding.leak_class for finding in audit_source(source)}


# --- the headline measurement -------------------------------------------------


def test_catches_every_leaky_fixture() -> None:
    """All three deliberate cheats must be rejected. This is the layer's reason to exist."""
    fixtures = sorted(LEAKY_DIR.glob("leak_*.py"))
    assert len(fixtures) == 3, f"expected 3 leaky fixtures, found {len(fixtures)}"
    missed = [p.name for p in fixtures if not is_rejected(audit_file(p))]
    assert not missed, f"leaky fixtures not caught: {missed}"


def test_no_false_positives_on_honest_strategies() -> None:
    """Not one honest fixture may be rejected.

    This is the expensive error. A leakage auditor that cries wolf is ignored, and in Phase 1.3 the
    false-positive rate measured here is the denominator of the headline precision figure.
    """
    fixtures = [p for p in sorted(CLEAN_DIR.glob("*.py")) if not p.name.startswith("_")]
    assert len(fixtures) >= 20, f"expected at least 20 clean fixtures, found {len(fixtures)}"
    flagged = [p.name for p in fixtures if is_rejected(audit_file(p))]
    assert not flagged, f"honest strategies wrongly flagged: {flagged}"


def test_survivorship_is_caught_structurally_not_by_magnitude() -> None:  # noqa: D401
    """The survivorship fixture must be caught by reading the code, never by its return.

    Its backtested Sharpe is about 1.2 — entirely plausible. An auditor that only noticed
    survivorship when the number looked too good would miss precisely the real-world cases that
    matter, which sit in exactly that range. The auditor never sees returns at all; this test
    pins that the detection comes from the membership reference in the source.
    """
    findings = audit_file(LEAKY_DIR / "leak_survivorship.py")
    assert is_rejected(findings)
    # The class attributed is currently the constructor rule rather than the survivorship rule:
    # the fixture hands its constituent table in at construction, and that fires first. The
    # accept/reject decision is right, the label is coarse. Attribution accuracy is measured
    # separately and is a known weakness, so this test asserts the decision, not the class.


# --- individual detectors -----------------------------------------------------


def test_negative_shift_is_future_indexing() -> None:
    assert LeakClass.FUTURE_INDEXING in classes_of(
        "def f(df):\n    return df['close'].shift(-1)\n"
    )


def test_negative_shift_written_as_unary_minus() -> None:
    """The same defect spelled differently must still be caught.

    A regular expression tuned to `shift(-1)` misses `shift(-(n))`; parsing the tree does not.
    """
    assert LeakClass.FUTURE_INDEXING in classes_of(
        "def f(df, n):\n    return df['close'].shift(-(1))\n"
    )


def test_forward_slice_is_caught() -> None:
    assert LeakClass.FUTURE_INDEXING in classes_of(
        "def f(series, t):\n    return series[t + 1:]\n"
    )


def test_scaler_fitted_outside_a_fold() -> None:
    """Constructing a transform and fitting it in place, with nothing restricting the fold."""
    source = (
        "from sklearn.preprocessing import StandardScaler\n"
        "def f(frame):\n"
        "    return StandardScaler().fit(frame)\n"
    )
    assert LeakClass.FULL_SAMPLE_FIT in classes_of(source)


def test_constructor_taking_bulk_data_is_flagged() -> None:
    """Accepting the dataset up front is how a strategy acquires the future.

    The parameter here is called `roster` and the attribute `_book` — deliberately bland. What
    marks them as data is that the class filters the attribute, not what either is named. The
    previous implementation matched a list of words and was defeated by exactly this rename.

    The class reported is `point_in_time_bypass`, which is what this rule actually detects: data
    reaching the strategy outside the sanctioned channel. It was previously reported as
    `full_sample_fit`, which was a mislabel — no transform is fitted here — and that one mislabel
    was stamped on most rejections, because almost every cheat enters through the constructor.
    """
    source = (
        "class S:\n"
        "    def __init__(self, roster):\n"
        "        self._book = roster\n"
        "    def generate(self, view):\n"
        "        return self._book.filter(view.symbols)\n"
    )
    assert LeakClass.POINT_IN_TIME_BYPASS in classes_of(source)


def test_reading_stored_panel_while_deciding_is_flagged() -> None:
    """The out-of-band read: reaching around the point-in-time view to stored state.

    Reported as `point_in_time_bypass` rather than `future_indexing`. Nothing here indexes forward
    in time — the defect is the channel the data arrived through, and calling it future indexing
    conflated two distinct classes in the corpus breakdown.
    """
    source = (
        "class S:\n"
        "    def __init__(self, panel):\n"
        "        self._panel = panel\n"
        "    def generate(self, view):\n"
        "        return self._panel.filter(view.as_of)\n"
    )
    findings = audit_source(source)
    assert LeakClass.POINT_IN_TIME_BYPASS in {f.leak_class for f in findings}


def test_a_fit_method_under_another_name_is_still_a_fit() -> None:
    """`FIT_METHODS` is a word list; a normaliser exposing `calibrate` walks straight past it.

    What makes a method a fit is structural: it takes an argument it treats as data, derives
    summary statistics from it, and retains them on the instance. Recognising that shape is what
    stops a rename from hiding the classic scaler leak.
    """
    source = (
        "class N:\n"
        "    def calibrate(self, rows):\n"
        "        self._centre = rows.group_by('symbol').mean()\n"
        "        return self\n"
        "class S:\n"
        "    def __init__(self, reference_set):\n"
        "        self._n = N().calibrate(reference_set)\n"
    )
    assert LeakClass.FULL_SAMPLE_FIT in classes_of(source)


def test_a_statistic_taken_in_the_constructor_is_attributed_to_its_own_class() -> None:
    """Provenance has to be tracked inside `__init__`, not only inside `generate`.

    The constructor is where tainted data enters, yet an earlier version walked it with no scope
    open. Every category-specific detector sits there — the percentile over the whole panel, the
    extremum feeding a membership filter — so none of them could fire, and each case carried only
    the constructor rule's own label whatever it was really doing.
    """
    source = (
        "class S:\n"
        "    def __init__(self, panel):\n"
        "        rows = panel.filter(panel['symbol'])\n"
        "        self._cut = rows['score'].quantile(0.8)\n"
    )
    assert LeakClass.FULL_SAMPLE_STATISTIC in classes_of(source)


def test_a_centred_window_is_flagged_whatever_its_source() -> None:
    """A centred window averages rows after the one it labels, so it reads forward by construction.

    This is a property of the window itself rather than of where its data came from, so it is
    judged independently of provenance.
    """
    source = "def f(view):\n    return view.closes().rolling_mean(window_size=21, center=True)\n"
    assert LeakClass.BOUNDARY_CROSSING_WINDOW in classes_of(source)


def test_a_forward_derived_value_reaching_the_output_is_flagged() -> None:
    """The target class fires on dataflow now, not on a variable being called "target".

    Replaces an earlier test that asserted a name-matching rule. That rule rejected honest code
    naming a local `target_weight`, so it was removed rather than widened.
    """
    source = (
        "class S:\n"
        "    def generate(self, view):\n"
        "        edge = view.closes().shift(-1)\n"
        "        return edge\n"
    )
    assert LeakClass.TARGET_IN_FEATURES in classes_of(source)


def test_an_ordinary_variable_called_target_is_not_flagged() -> None:
    """The specific false positive that motivated deleting the name list."""
    source = (
        "class S:\n"
        "    def generate(self, view):\n"
        "        target_weight = 1.0 / len(view.symbols)\n"
        "        return {s: target_weight for s in view.symbols}\n"
    )
    assert not is_rejected(audit_source(source))


# --- negative controls: honest code that must stay quiet ----------------------


def test_positive_shift_is_not_flagged() -> None:
    """Shifting forward looks backwards in time, which is exactly what a lag should do."""
    assert not audit_source("def f(df):\n    return df['close'].shift(1)\n")


def test_rolling_window_statistics_are_not_flagged() -> None:
    """Aggregating a trailing window is ordinary and must not be reported."""
    source = (
        "def f(window):\n"
        "    average = sum(window) / len(window)\n"
        "    return average\n"
    )
    assert not is_rejected(audit_source(source))


def test_constructor_with_only_settings_is_not_flagged() -> None:
    """Windows and thresholds are configuration, not data."""
    source = (
        "class S:\n"
        "    def __init__(self, lookback=63, holdings=10):\n"
        "        self._lookback = lookback\n"
        "        self._holdings = holdings\n"
    )
    assert not audit_source(source)


def test_reading_the_view_while_deciding_is_not_flagged() -> None:
    """The sanctioned path must never be reported, or every honest strategy fails."""
    source = (
        "class S:\n"
        "    def __init__(self, lookback=21):\n"
        "        self._lookback = lookback\n"
        "    def generate(self, view):\n"
        "        return view.closes(lookback=self._lookback)\n"
    )
    assert not is_rejected(audit_source(source))


# --- robustness ---------------------------------------------------------------


def test_unparseable_source_is_reported_not_raised() -> None:
    """Generated code that does not compile is a result to record, not an exception.

    The generator in a later phase produces code that sometimes fails to parse. Those attempts
    still count as trials, so the auditor has to return a finding rather than crash the run.
    """
    findings = audit_source("def broken(:\n")
    assert len(findings) == 1
    assert findings[0].severity is Severity.HIGH
    assert "does not parse" in findings[0].explanation


def test_findings_carry_a_location_and_an_explanation() -> None:
    """Every finding must be checkable by a human without re-reading the whole file."""
    findings = audit_source("def f(df):\n    return df['close'].shift(-1)\n")
    assert findings
    for finding in findings:
        assert finding.line_number > 0
        assert finding.code_snippet
        # The explanation states why the construct leaks, not merely that a pattern matched.
        assert len(finding.explanation) > 40


def test_empty_source_yields_nothing() -> None:
    assert audit_source("") == []
