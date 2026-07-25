"""Tests for the statistical deflation layer.

Three groups, and they are not equally load-bearing.

The Sharpe deflation tests are mostly *direction-of-effect* tests. There is no external oracle for
a Deflated Sharpe Ratio, so the defence against a transcription error in the formula is that every
input moves the answer the way the theory says it must: more trials must deflate harder, a longer
sample must sharpen the estimate, negative skew and fat tails must weaken it. A sign error anywhere
flips one of these. Two known-answer anchors pin the level: at the luck threshold the DSR is
exactly one half, and one worked example is frozen so the number quoted in the checkpoint report
cannot drift silently.

The PBO tests check calibration against two cases where the answer is known by construction: pure
noise, where selection carries no information and the expected PBO is one half, and a corpus
containing one genuinely dominant strategy, where it is zero.

The trial-counter tests are the ones that matter most. The counter is the input nobody can audit
from the output — a Deflated Sharpe computed against an understated N looks exactly as respectable
as an honest one. So they are written adversarially: it is not enough that a clean ledger verifies,
the tampering has to be caught, including the tampering that tries to cover its tracks.
"""

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.audit.stat import (
    LEDGER_GENESIS,
    MAX_PARTITIONS,
    TamperError,
    TrialCounter,
    _entry_digest,  # private on purpose; used below to forge a plausible tampering attempt
    default_counter_path,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from src.common.config import Config
from src.common.exceptions import LabError

# Sessions per year used to convert the annualised figures a human quotes into the
# per-observation units the module works in. Deliberately spelled out at every call site below
# rather than hidden in a helper: mixing frequencies is the easiest way to inflate a DSR.
SESSIONS_PER_YEAR = 252


# --- expected maximum Sharpe under pure luck ------------------------------------


def test_expected_max_sharpe_rises_with_the_number_of_trials() -> None:
    """Search harder and the bar for "impressive" rises. This is the whole idea of deflation."""
    thresholds = [expected_max_sharpe(n, variance_of_trial_sharpes=1.0) for n in (2, 10, 100, 1000)]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] < thresholds[-1]


def test_expected_max_sharpe_rises_with_the_dispersion_of_the_search() -> None:
    """Two searches of equal size are not equally impressive if one ranged over wilder strategies.

    The threshold scales with sqrt(V), so quadrupling the variance must double it.
    """
    narrow = expected_max_sharpe(200, variance_of_trial_sharpes=0.25)
    wide = expected_max_sharpe(200, variance_of_trial_sharpes=1.00)
    assert wide == pytest.approx(2.0 * narrow)


def test_a_single_trial_deflates_nothing() -> None:
    """One trial is not a search. The asymptotic formula is undefined at N=1; the answer is zero."""
    assert expected_max_sharpe(1, variance_of_trial_sharpes=1.0) == 0.0


def test_zero_dispersion_deflates_nothing() -> None:
    """If every trial produced the identical Sharpe there was no luck to select on."""
    assert expected_max_sharpe(500, variance_of_trial_sharpes=0.0) == 0.0


@pytest.mark.parametrize("n_trials", [0, -1, -100])
def test_expected_max_sharpe_rejects_a_non_positive_trial_count(n_trials: int) -> None:
    with pytest.raises(ValueError, match="n_trials must be at least 1"):
        expected_max_sharpe(n_trials, variance_of_trial_sharpes=1.0)


def test_expected_max_sharpe_rejects_a_negative_variance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        expected_max_sharpe(100, variance_of_trial_sharpes=-0.5)


# --- the Probabilistic Sharpe Ratio ---------------------------------------------


def test_psr_is_one_half_when_the_observed_sharpe_equals_the_benchmark() -> None:
    """No evidence either way is exactly a coin flip, whatever the moments happen to be."""
    psr = probabilistic_sharpe_ratio(0.1, benchmark_sharpe=0.1, n_observations=500, skew=-0.4,
                                     kurtosis=6.0)
    assert psr == pytest.approx(0.5)


def test_psr_rises_with_the_length_of_the_sample() -> None:
    """The same Sharpe measured over more observations is stronger evidence of the same claim."""
    values = [
        probabilistic_sharpe_ratio(0.05, 0.0, n_observations=n, skew=0.0, kurtosis=3.0)
        for n in (50, 250, 1000, 5000)
    ]
    assert values == sorted(values)


def test_negative_skew_weakens_the_same_sharpe() -> None:
    """A Sharpe earned by selling tail risk is weaker evidence than one earned symmetrically.

    This is the correction a normality-assuming t-test cannot make, and getting the sign wrong
    would quietly reward exactly the return profile that ought to be penalised.
    """
    symmetric = probabilistic_sharpe_ratio(0.05, 0.0, 1000, skew=0.0, kurtosis=3.0)
    left_tailed = probabilistic_sharpe_ratio(0.05, 0.0, 1000, skew=-1.5, kurtosis=3.0)
    right_tailed = probabilistic_sharpe_ratio(0.05, 0.0, 1000, skew=1.5, kurtosis=3.0)
    assert left_tailed < symmetric < right_tailed


def test_fatter_tails_weaken_the_same_sharpe() -> None:
    """Kurtosis here is non-excess, so 3.0 is the normal case and larger means fatter."""
    values = [
        probabilistic_sharpe_ratio(0.05, 0.0, 1000, skew=0.0, kurtosis=k)
        for k in (3.0, 6.0, 12.0, 30.0)
    ]
    assert values == sorted(values, reverse=True)


def test_psr_rejects_excess_kurtosis_passed_by_mistake() -> None:
    """Excess kurtosis of 0.0 means a normal distribution, but the formula wants 3.0.

    Passing the wrong convention shrinks the standard error and flatters the result, so it is
    refused rather than silently accepted.
    """
    with pytest.raises(ValueError, match="non-excess"):
        probabilistic_sharpe_ratio(0.05, 0.0, 1000, skew=0.0, kurtosis=0.0)


@pytest.mark.parametrize("n_observations", [0, 1, -5])
def test_psr_rejects_a_sample_too_short_to_estimate_anything(n_observations: int) -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        probabilistic_sharpe_ratio(0.05, 0.0, n_observations, skew=0.0, kurtosis=3.0)


def test_psr_refuses_an_undefined_standard_error_rather_than_returning_nan() -> None:
    """Extreme skew against a large Sharpe drives the variance term non-positive.

    The standard error would be the square root of a negative number. Returning nan here would be
    the worst outcome: it propagates and eventually reads as an ordinary-looking probability.
    """
    with pytest.raises(ValueError, match="non-positive variance term"):
        probabilistic_sharpe_ratio(1.0, 0.0, 1000, skew=5.0, kurtosis=3.0)


# --- the Deflated Sharpe Ratio --------------------------------------------------


def test_dsr_is_one_half_when_the_observed_sharpe_is_exactly_the_luck_threshold() -> None:
    """The known-answer anchor. A result no better than the search would produce anyway is 50/50.

    Computed by asking :func:`expected_max_sharpe` for the threshold and handing that same figure
    back as the observed Sharpe. Anything other than 0.5 means the DSR is not centring on the
    threshold it claims to.
    """
    n_trials, variance = 200, 1.0 / SESSIONS_PER_YEAR
    threshold = expected_max_sharpe(n_trials, variance)
    dsr = deflated_sharpe_ratio(
        observed_sharpe=threshold,
        n_trials=n_trials,
        n_observations=2520,
        skew=-0.5,
        kurtosis=5.0,
        variance_of_trial_sharpes=variance,
    )
    assert dsr == pytest.approx(0.5)


def test_dsr_decreases_strictly_as_the_trial_count_rises() -> None:
    """Everything else held fixed, trying more things must make the same result less impressive.

    The single most important monotonicity in the module. If this ever failed, the deflation would
    be rewarding a wider search instead of penalising it, and every headline number in P1 would be
    biased in the flattering direction.
    """
    # Annotated because the mapping mixes an int observation count with float moments; without
    # it the inferred value type collapses to float and the unpacked call fails type checking.
    kwargs: dict[str, Any] = {
        "observed_sharpe": 2.0 / math.sqrt(SESSIONS_PER_YEAR),
        "n_observations": 2520,
        "skew": -0.5,
        "kurtosis": 5.0,
        "variance_of_trial_sharpes": 1.0 / SESSIONS_PER_YEAR,
    }
    values = [deflated_sharpe_ratio(n_trials=n, **kwargs) for n in (1, 2, 10, 50, 200, 1000, 5000)]
    for stricter, looser in zip(values[1:], values[:-1], strict=True):
        assert stricter < looser


def test_dsr_never_exceeds_the_undeflated_psr() -> None:
    """Deflation can only ever subtract confidence, never add it."""
    args: dict[str, Any] = {"n_observations": 1500, "skew": -0.3, "kurtosis": 7.0}
    psr = probabilistic_sharpe_ratio(0.06, 0.0, **args)
    dsr = deflated_sharpe_ratio(0.06, n_trials=300, variance_of_trial_sharpes=0.002, **args)
    assert dsr <= psr


@pytest.mark.parametrize("n_trials", [1, 2, 25, 300, 10_000])
def test_dsr_is_always_a_probability(n_trials: int) -> None:
    """It is consumed downstream as a probability, so the range is part of the contract."""
    dsr = deflated_sharpe_ratio(0.08, n_trials, 2520, skew=-1.0, kurtosis=9.0,
                                variance_of_trial_sharpes=0.003)
    assert 0.0 <= dsr <= 1.0
    assert not math.isnan(dsr)


# The worked example quoted in the Checkpoint 1.3 report, frozen so it cannot drift.
#
# Scenario: a strategy showing an annualised Sharpe of 2.0 over ten years of daily data
# (2520 observations), found after 200 trials, with modestly non-normal daily returns
# (skewness -0.5, non-excess kurtosis 5.0).
#
# Everything is converted to per-observation units before it reaches the module, which is the
# conversion a careless caller skips: SR_daily = 2.0 / sqrt(252), V_daily = V_annual / 252.
#
# Two dispersions are recorded rather than one, because the DSR's sensitivity to V is the honest
# caveat on the whole method and hiding it behind a single chosen value would misrepresent it. V
# is the variance of Sharpe ratios *across the 200 trials*; it is measured from the corpus, not
# assumed, and these two values bracket a plausible range for a heterogeneous generated corpus.
WORKED_EXAMPLE_ANNUAL_SHARPE = 2.0
WORKED_EXAMPLE_OBSERVATIONS = 2520
WORKED_EXAMPLE_TRIALS = 200
WORKED_EXAMPLE_SKEW = -0.5
WORKED_EXAMPLE_KURTOSIS = 5.0
WORKED_EXAMPLE_CASES = [
    # (annualised variance of trial Sharpes, expected DSR)
    (0.25, 0.96986468),   # a narrow search: trial Sharpes with an annualised sd of 0.5
    (1.00, 0.00989844),   # a wide search: trial Sharpes with an annualised sd of 1.0
]


@pytest.mark.parametrize(("annual_trial_variance", "expected_dsr"), WORKED_EXAMPLE_CASES)
def test_worked_example_for_the_checkpoint_report(
    annual_trial_variance: float, expected_dsr: float
) -> None:
    """Freeze the figures the checkpoint report quotes, so a refactor cannot move them quietly.

    The pair of results is itself the finding: the same raw Sharpe of 2.0 deflates to 0.97 under a
    narrow search and to 0.01 under a wide one. The DSR is not a property of the strategy alone.
    """
    dsr = deflated_sharpe_ratio(
        observed_sharpe=WORKED_EXAMPLE_ANNUAL_SHARPE / math.sqrt(SESSIONS_PER_YEAR),
        n_trials=WORKED_EXAMPLE_TRIALS,
        n_observations=WORKED_EXAMPLE_OBSERVATIONS,
        skew=WORKED_EXAMPLE_SKEW,
        kurtosis=WORKED_EXAMPLE_KURTOSIS,
        variance_of_trial_sharpes=annual_trial_variance / SESSIONS_PER_YEAR,
    )
    assert dsr == pytest.approx(expected_dsr, abs=5e-8)


def test_worked_example_looks_certain_before_deflation() -> None:
    """The counterpart that makes the example worth quoting.

    Undeflated, the same strategy's PSR against zero rounds to 1.0 at every dispersion. All of the
    doubt in the wide-search case above comes from the trial count, none of it from the sample.
    """
    psr = probabilistic_sharpe_ratio(
        observed_sharpe=WORKED_EXAMPLE_ANNUAL_SHARPE / math.sqrt(SESSIONS_PER_YEAR),
        benchmark_sharpe=0.0,
        n_observations=WORKED_EXAMPLE_OBSERVATIONS,
        skew=WORKED_EXAMPLE_SKEW,
        kurtosis=WORKED_EXAMPLE_KURTOSIS,
    )
    assert psr > 0.999


# --- Probability of Backtest Overfitting ----------------------------------------


def _noise_matrix(
    rng: np.random.Generator, n_observations: int = 1000, n_strategies: int = 20
) -> np.ndarray:
    """Independent zero-mean daily returns: no strategy here has any edge at all."""
    return rng.normal(loc=0.0, scale=0.01, size=(n_observations, n_strategies))


def test_pbo_is_one_half_for_pure_noise() -> None:
    """Calibration against a case whose answer is known by construction.

    Train and test blocks are disjoint, so for independent returns a strategy's in-sample and
    out-of-sample means are independent. Selecting the in-sample winner therefore says nothing
    about its out-of-sample rank, the rank is uniform, and the expected PBO is one half.

    Averaged over replications rather than asserted on one draw, because a single draw is a noisy
    estimate of that expectation — see the test below, which pins the noise down deliberately. The
    tolerance is roughly three standard errors of the mean at this replication count.
    """
    rng = np.random.default_rng(42)
    values = [probability_of_backtest_overfitting(_noise_matrix(rng), n_splits=16)
              for _ in range(40)]
    assert np.mean(values) == pytest.approx(0.5, abs=0.10)


def test_a_single_pbo_estimate_is_far_noisier_than_its_mean() -> None:
    """Recorded as a caveat, not a nicety: one PBO number carries a wide error bar.

    Across replications on identically-distributed noise the statistic ranges over most of the unit
    interval. Anyone reading a single strategy corpus's PBO as a precise quantity is overreading
    it, and this test exists so that fact is written down where it cannot be forgotten.
    """
    rng = np.random.default_rng(42)
    values = [probability_of_backtest_overfitting(_noise_matrix(rng), n_splits=16)
              for _ in range(40)]
    assert np.std(values) > 0.10
    assert max(values) - min(values) > 0.30


def test_pbo_is_near_zero_when_one_strategy_genuinely_dominates() -> None:
    """The other end of the scale. A persistent edge keeps winning on data it was not chosen on.

    Strategy 3 is given a real positive drift on top of the same noise, so it is picked in-sample
    and stays top out-of-sample. If this returned something near 0.5 the statistic would be
    incapable of distinguishing a real edge from selection noise.
    """
    rng = np.random.default_rng(42)
    returns = _noise_matrix(rng)
    returns[:, 3] += 0.004
    assert probability_of_backtest_overfitting(returns, n_splits=16) < 0.05


def test_a_flat_strategy_cannot_win_the_in_sample_selection() -> None:
    """A zero-variance series has no Sharpe; it must score zero, not divide through noise.

    Without the guard, a constant column's Sharpe is its mean over a floating-point residue near
    1e-16, which is astronomically large, and it would be selected in every single partition.
    """
    rng = np.random.default_rng(42)
    returns = _noise_matrix(rng)
    returns[:, 0] = 0.001  # perfectly flat, and positive
    # The dominant column below is what should be selected instead.
    returns[:, 3] += 0.004
    assert probability_of_backtest_overfitting(returns, n_splits=16) < 0.05


@pytest.mark.parametrize("n_splits", [4, 8, 12, 16])
def test_pbo_is_a_probability_at_every_supported_split_count(n_splits: int) -> None:
    rng = np.random.default_rng(42)
    pbo = probability_of_backtest_overfitting(_noise_matrix(rng), n_splits=n_splits)
    assert 0.0 <= pbo <= 1.0


def test_pbo_rejects_a_one_dimensional_series() -> None:
    """A single series has no peers to rank against, so the statistic is undefined."""
    with pytest.raises(ValueError, match="must be 2-D"):
        probability_of_backtest_overfitting(np.zeros(500), n_splits=16)


def test_pbo_rejects_a_single_strategy() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        probability_of_backtest_overfitting(np.zeros((500, 1)), n_splits=16)


@pytest.mark.parametrize("n_splits", [3, 15, 1, 0, -4])
def test_pbo_rejects_an_odd_or_degenerate_split_count(n_splits: int) -> None:
    """The partitions must be balanced, which requires an even number of blocks."""
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="even number"):
        probability_of_backtest_overfitting(_noise_matrix(rng), n_splits=n_splits)


def test_pbo_rejects_a_series_too_short_for_the_requested_splits() -> None:
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="need at least"):
        probability_of_backtest_overfitting(_noise_matrix(rng, n_observations=20), n_splits=16)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_pbo_rejects_non_finite_returns(bad: float) -> None:
    """A nan would propagate through the argmax and produce a plausible-looking number."""
    rng = np.random.default_rng(42)
    returns = _noise_matrix(rng)
    returns[100, 2] = bad
    with pytest.raises(ValueError, match="nan or inf"):
        probability_of_backtest_overfitting(returns, n_splits=16)


def test_pbo_refuses_a_split_count_that_would_take_all_day() -> None:
    """32 splits enumerate 601 million partitions. Raise rather than hang.

    The failure mode this guards against is not a crash; it is a run that appears to be working
    for hours because someone raised n_splits without checking what it costs.
    """
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="above the"):
        probability_of_backtest_overfitting(
            _noise_matrix(rng, n_observations=2000), n_splits=32
        )
    assert math.comb(32, 16) > MAX_PARTITIONS


def test_pbo_refuses_a_corpus_too_wide_to_hold_in_memory() -> None:
    """The working grid is (partitions x strategies); at 16 splits that caps the corpus width."""
    rng = np.random.default_rng(42)
    wide = _noise_matrix(rng, n_observations=64, n_strategies=2000)
    with pytest.raises(ValueError, match="working-memory ceiling"):
        probability_of_backtest_overfitting(wide, n_splits=16)


# --- the tamper-evident trial counter -------------------------------------------


def _ledger(tmp_path: Path) -> TrialCounter:
    return TrialCounter(tmp_path / "nested" / "trial_ledger.jsonl")


def test_a_ledger_that_does_not_exist_yet_is_an_honest_zero(tmp_path: Path) -> None:
    """No trials recorded is a legitimate state, not an error.

    Worth stating what this does *not* cover: deleting the ledger outright is indistinguishable
    from never having created one. Nothing in-band can fix that, which is why the location matters
    (see the last test in this file).
    """
    counter = TrialCounter(tmp_path / "never_written.jsonl")
    assert counter.count() == 0
    assert counter.verify() == 0


def test_record_returns_the_running_total(tmp_path: Path) -> None:
    counter = _ledger(tmp_path)
    assert counter.record("sma_crossover", "evaluated") == 1
    assert counter.record("momentum_63d", "evaluated") == 2
    assert counter.count() == 2


def test_failures_and_syntax_errors_count_as_trials(tmp_path: Path) -> None:
    """The rule the whole deflation rests on.

    A generator that logged only the strategies that ran would report N=2 here instead of N=5.
    Understating N lowers the luck threshold, which raises the deflated Sharpe of every survivor —
    a systematic bias in the flattering direction, invisible in the output.
    """
    counter = _ledger(tmp_path)
    counter.record("candidate_001", "evaluated")
    counter.record("candidate_002", "syntax_error")
    counter.record("candidate_003", "runtime_error")
    counter.record("candidate_004", "syntax_error")
    counter.record("candidate_005", "evaluated")

    assert counter.count() == 5
    assert counter.verify() == 5

    outcomes = [json.loads(line)["outcome"]
                for line in counter.path.read_text(encoding="utf-8").splitlines()]
    assert outcomes.count("evaluated") == 2
    assert outcomes.count("syntax_error") == 2
    assert outcomes.count("runtime_error") == 1


def test_the_first_entry_chains_from_the_fixed_genesis(tmp_path: Path) -> None:
    """Pinning the head of the chain is what stops the opening entries being dropped wholesale."""
    counter = _ledger(tmp_path)
    counter.record("first", "evaluated")
    first = json.loads(counter.path.read_text(encoding="utf-8").splitlines()[0])

    assert first["prev_hash"] == hashlib.sha256(LEDGER_GENESIS.encode("utf-8")).hexdigest()
    assert first["seq"] == 1


def test_each_entry_chains_to_the_previous_one(tmp_path: Path) -> None:
    counter = _ledger(tmp_path)
    for i in range(5):
        counter.record(f"candidate_{i}", "evaluated")

    entries = [json.loads(line) for line in counter.path.read_text(encoding="utf-8").splitlines()]
    for earlier, later in zip(entries[:-1], entries[1:], strict=True):
        assert later["prev_hash"] == earlier["hash"]


def test_a_second_counter_on_the_same_path_continues_the_chain(tmp_path: Path) -> None:
    """The ledger lives on disk, not in the process. Restarting a run must not restart N."""
    path = tmp_path / "trial_ledger.jsonl"
    TrialCounter(path).record("before_restart", "evaluated")
    resumed = TrialCounter(path)
    assert resumed.record("after_restart", "evaluated") == 2
    assert resumed.verify() == 2


@pytest.mark.parametrize("outcome", ["success", "ok", "", "EVALUATED", "skipped"])
def test_record_rejects_an_unrecognised_outcome(tmp_path: Path, outcome: str) -> None:
    """Outcome classes are fixed. A free-text outcome makes the failure breakdown unanalysable."""
    with pytest.raises(ValueError, match="outcome must be one of"):
        # Deliberately passing a value outside the accepted literals: the point of the test is
        # that the check exists at runtime, since the ledger is also written by other processes
        # that a type checker never sees.
        _ledger(tmp_path).record("candidate", outcome)  # type: ignore[arg-type]


def test_record_rejects_an_anonymous_trial(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _ledger(tmp_path).record("", "evaluated")


def _rewrite_line(
    counter: TrialCounter,
    index: int,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """Hand-edit one ledger line, exactly as somebody with a text editor would."""
    lines = counter.path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[index])
    mutate(entry)
    lines[index] = json.dumps(entry, sort_keys=True)
    counter.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _five_trials(tmp_path: Path) -> TrialCounter:
    counter = _ledger(tmp_path)
    for i in range(5):
        counter.record(f"candidate_{i}", "evaluated" if i % 2 == 0 else "syntax_error")
    assert counter.verify() == 5
    return counter


def test_editing_a_middle_entry_breaks_verification(tmp_path: Path) -> None:
    """The headline requirement. Rewrite line 3's payload and the chain must not reconstruct."""
    counter = _five_trials(tmp_path)
    _rewrite_line(counter, 2, lambda e: e.update(strategy_name="something_else"))

    with pytest.raises(TamperError, match="line 3"):
        counter.verify()


def test_relabelling_a_failure_as_a_success_is_caught(tmp_path: Path) -> None:
    """The specific edit a dishonest run would want to make, so it gets its own test."""
    counter = _five_trials(tmp_path)
    _rewrite_line(counter, 1, lambda e: e.update(outcome="evaluated"))

    with pytest.raises(TamperError, match="edited"):
        counter.verify()


def test_covering_the_edit_by_recomputing_that_line_only_still_fails(tmp_path: Path) -> None:
    """A tamperer who knows about the hash will fix the line they touched. That is not enough.

    Recomputing entry 3's own digest makes entry 3 self-consistent, but entry 4 still records the
    old digest as its ``prev_hash``, so the break simply moves one line down. This is the property
    that makes it a chain rather than a set of independent checksums.
    """
    counter = _five_trials(tmp_path)

    def forge(entry: dict[str, Any]) -> None:
        entry["strategy_name"] = "something_else"
        entry["hash"] = _entry_digest(entry["prev_hash"], entry)

    _rewrite_line(counter, 2, forge)

    with pytest.raises(TamperError, match="line 4"):
        counter.verify()


def test_appending_after_tampering_does_not_repair_the_ledger(tmp_path: Path) -> None:
    """New honest entries do not launder an earlier edit. The break stays where it was made."""
    counter = _five_trials(tmp_path)
    _rewrite_line(counter, 2, lambda e: e.update(strategy_name="something_else"))

    counter.record("candidate_5", "evaluated")
    counter.record("candidate_6", "evaluated")
    assert counter.count() == 7

    with pytest.raises(TamperError, match="line 3"):
        counter.verify()


def test_deleting_an_entry_is_caught(tmp_path: Path) -> None:
    """Deletion is the tampering with the strongest motive: it lowers N and inflates every DSR."""
    counter = _five_trials(tmp_path)
    lines = counter.path.read_text(encoding="utf-8").splitlines()
    del lines[2]
    counter.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TamperError):
        counter.verify()


def test_reordering_entries_is_caught(tmp_path: Path) -> None:
    counter = _five_trials(tmp_path)
    lines = counter.path.read_text(encoding="utf-8").splitlines()
    lines[1], lines[3] = lines[3], lines[1]
    counter.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TamperError):
        counter.verify()


def test_a_corrupt_line_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """Truncated writes and half-flushed lines must fail loudly, not be quietly dropped."""
    counter = _five_trials(tmp_path)
    with open(counter.path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 6, "strategy_name": "trunc\n')

    with pytest.raises(TamperError, match="not valid JSON"):
        counter.verify()


def test_an_entry_missing_a_field_is_reported(tmp_path: Path) -> None:
    counter = _five_trials(tmp_path)
    _rewrite_line(counter, 1, lambda e: e.pop("hash"))

    with pytest.raises(TamperError, match="missing fields"):
        counter.verify()


def test_an_unknown_outcome_on_disk_is_reported(tmp_path: Path) -> None:
    """Validating at write time is not enough; the file can be edited after the fact."""
    counter = _five_trials(tmp_path)
    _rewrite_line(counter, 1, lambda e: e.update(outcome="probably_fine"))

    with pytest.raises(TamperError, match="unknown outcome"):
        counter.verify()


def test_tamper_error_is_a_lab_error() -> None:
    """So the top-level handler that catches LabError catches this too."""
    assert issubclass(TamperError, LabError)


def test_default_counter_path_sits_under_processed_data(
    config_with_runs_in_tmp: Config,
) -> None:
    """Outside the generator's working area, which is the only thing making the ledger credible.

    The hash chain detects edits by anything that cannot rewrite the whole file. It does not defend
    against a process with full write access, so the location is part of the control, not an
    implementation detail.
    """
    path = default_counter_path(config_with_runs_in_tmp)
    assert path.name == "trial_ledger.jsonl"
    assert path.parent == config_with_runs_in_tmp.paths.data_processed
