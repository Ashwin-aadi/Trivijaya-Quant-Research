"""Statistical deflation: how much of an apparent edge is selection luck rather than skill.

Single responsibility — given a reported performance figure and an honest count of how many
strategies were tried to obtain it, say how much of that figure survives.

Three instruments live here.

**Probabilistic and Deflated Sharpe Ratios.** Bailey & Lopez de Prado, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality", *Journal of Portfolio
Management* 40(5), pp. 94-107, 2014. The DSR asks: given that N strategies were tried, and given
that this one's returns are skewed, fat-tailed and measured over a finite sample, what is the
probability its true Sharpe is above zero?

**Probability of Backtest Overfitting**, via Combinatorially Symmetric Cross-Validation. Bailey,
Borwein, Lopez de Prado & Zhu, "The Probability of Backtest Overfitting", *Journal of Computational
Finance* 20(4), pp. 39-69, 2016. A different question: across many train/test splits of the same
history, how often does the strategy that looked best in-sample land below the median out-of-sample?

**A tamper-evident trial counter.** Both of the above are functions of N, the number of strategies
tried. N is the single input a researcher has every incentive to understate, and understating it is
undetectable from the output alone — a deflated Sharpe computed from N=5 when the truth was N=300
looks exactly as respectable as an honest one. So N is not a number someone remembers and passes
in; it is read off an append-only, hash-chained ledger that records every attempt, successes and
failures alike.

**Units.** Every Sharpe ratio in this module is a *per-observation* Sharpe, measured at the same
sampling frequency as ``n_observations``. Passing an annualised Sharpe alongside a daily
observation count is the easiest mistake to make here and it inflates the answer badly: the
observed figure grows by sqrt(252) while the luck threshold and the standard error do not scale
with it in the same way. Convert first — ``SR_daily = SR_annual / sqrt(252)``, and likewise
``V_daily = V_annual / 252`` for the trial-Sharpe variance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

from src.common.config import Config
from src.common.exceptions import LabError

# Euler-Mascheroni constant, the `gamma` in the expected-maximum expression below.
EULER_MASCHERONI = 0.5772156649015329

# CSCV enumerates C(n_splits, n_splits/2) partitions: 12,870 at the default 16 splits, 184,756 at
# 20, and 2.7 million at 24. Past this ceiling the call is refused rather than left to grind — a
# caller who wants more partitions should say so deliberately, not discover it as a hang.
MAX_PARTITIONS = 200_000

# The vectorised CSCV holds several (n_partitions, n_strategies) float64 grids at once. This bound
# keeps the largest of them around 160 MB, which is the point past which a consumer machine starts
# swapping instead of computing.
MAX_GRID_CELLS = 20_000_000

# Below this, a return series has no variation worth dividing by and its Sharpe is undefined.
# Matches the reasoning in src/eval/metrics.py: a constant series leaves floating-point residue
# near 1e-16, and dividing by residue manufactures an enormous Sharpe out of nothing.
NEGLIGIBLE_STD = 1e-12


# --- Deflated and Probabilistic Sharpe Ratios -----------------------------------


def expected_max_sharpe(n_trials: int, variance_of_trial_sharpes: float) -> float:
    """Sharpe the *best* of ``n_trials`` skill-free strategies is expected to show by luck alone.

    Bailey & Lopez de Prado (2014), eq. (5) — the expected maximum of N independent draws from a
    zero-mean normal with variance V:

        E[max SR] ~= sqrt(V) * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    where ``g`` is the Euler-Mascheroni constant and ``Z^-1`` the inverse standard normal CDF.

    ``variance_of_trial_sharpes`` is the variance of the Sharpe ratios *across trials* — the
    dispersion of the search, not of the returns. This is the term people omit, and omitting it
    matters: searching a wide, varied pool of strategies raises the bar for what counts as
    impressive far more than searching a narrow one with the same trial count.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if variance_of_trial_sharpes < 0.0:
        raise ValueError(
            f"variance_of_trial_sharpes must be non-negative, got {variance_of_trial_sharpes}"
        )
    if n_trials == 1:
        # The asymptotic expression is undefined at N=1 (Z^-1(1 - 1/1) = Z^-1(0) = -inf), but the
        # exact answer needs no approximation: the maximum of a single zero-mean draw has
        # expectation zero. One trial is not a search, so there is nothing to deflate away.
        return 0.0

    quantile_n = float(norm.ppf(1.0 - 1.0 / n_trials))
    quantile_ne = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    weighted = (1.0 - EULER_MASCHERONI) * quantile_n + EULER_MASCHERONI * quantile_ne
    return math.sqrt(variance_of_trial_sharpes) * weighted


def _sharpe_standard_error(
    observed_sharpe: float,
    n_observations: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Standard error of an estimated Sharpe when returns are not normal.

    Bailey & Lopez de Prado (2014), eq. (1), after Mertens (2002):

        sigma(SR_hat) = sqrt( (1 - g3*SR + (g4 - 1)/4 * SR^2) / (n - 1) )

    ``g3`` is skewness and ``g4`` is **non-excess** kurtosis — 3.0 for a normal distribution, not
    0.0. Negative skew and fat tails both widen this interval, which is the whole point: the same
    Sharpe earned by selling tail risk is weaker evidence of skill than one earned on well-behaved
    returns, and a normality-assuming test cannot tell the two apart.
    """
    if n_observations < 2:
        raise ValueError(f"need at least 2 observations to estimate a Sharpe, got {n_observations}")
    if kurtosis < 1.0:
        # Kurtosis here is non-excess and is bounded below by 1 for any real distribution. A value
        # under 1 almost always means excess kurtosis was passed by mistake; catch it rather than
        # let it shrink the standard error and flatter the result.
        raise ValueError(
            f"kurtosis is non-excess (3.0 for a normal) and cannot be below 1.0, got {kurtosis}"
        )

    variance_term = 1.0 - skew * observed_sharpe + 0.25 * (kurtosis - 1.0) * observed_sharpe**2
    if variance_term <= 0.0:
        # Reachable for extreme skew/Sharpe combinations. The standard error would be imaginary;
        # refuse rather than emit a nan that later reads as a perfectly ordinary probability.
        raise ValueError(
            f"non-positive variance term {variance_term:.6g} from skew={skew}, "
            f"kurtosis={kurtosis}, sharpe={observed_sharpe}; the standard error is undefined"
        )
    return math.sqrt(variance_term / (n_observations - 1))


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``, in [0, 1].

    Bailey & Lopez de Prado (2014), eq. (2):

        PSR(SR*) = Z[ (SR_hat - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) ]

    All Sharpe figures must be per-observation at the frequency of ``n_observations``; see the
    module docstring. This is the un-deflated statistic: it corrects for sample length and
    non-normality but knows nothing about how many strategies were tried.
    """
    standard_error = _sharpe_standard_error(observed_sharpe, n_observations, skew, kurtosis)
    return float(norm.cdf((observed_sharpe - benchmark_sharpe) / standard_error))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skew: float,
    kurtosis: float,
    variance_of_trial_sharpes: float,
) -> float:
    """Probability the true Sharpe exceeds zero after correcting for selection, in [0, 1].

    Bailey & Lopez de Prado (2014), eq. (6): the DSR is the PSR measured not against zero but
    against ``SR_0``, the Sharpe the luckiest of ``n_trials`` skill-free strategies would be
    expected to reach anyway.

    Reading it: 0.95 means the strategy would still look good had the search been accounted for;
    0.30 means what was found is roughly what searching that hard produces from noise. A raw
    Sharpe reported without this number is a statement about the search, not about the strategy.
    """
    luck_threshold = expected_max_sharpe(n_trials, variance_of_trial_sharpes)
    return probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=luck_threshold,
        n_observations=n_observations,
        skew=skew,
        kurtosis=kurtosis,
    )


# --- Probability of Backtest Overfitting (CSCV) ---------------------------------


def _block_moments(returns_matrix: np.ndarray, n_splits: int) -> tuple[np.ndarray, ...]:
    """Per-block observation count, sum, and sum of squares for every strategy.

    Reducing each block to three moments up front is what keeps CSCV cheap. A partition's Sharpe
    then costs a sum over ``n_splits/2`` short rows instead of a fresh pass over the return matrix,
    and there are 12,870 partitions to price at the default split count.

    ``np.array_split`` leaves blocks differing in length by at most one observation when the series
    does not divide evenly. The paper specifies equal submatrices; keeping every observation is
    preferred here to truncating the tail, and since the statistic compared across strategies is a
    length-normalised Sharpe the one-row difference cannot favour any strategy over another.
    """
    blocks = np.array_split(returns_matrix, n_splits, axis=0)
    counts = np.array([float(block.shape[0]) for block in blocks])
    sums = np.stack([block.sum(axis=0) for block in blocks])
    sums_of_squares = np.stack([np.square(block).sum(axis=0) for block in blocks])
    return counts, sums, sums_of_squares


def _sharpe_grid(
    mask: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sums_of_squares: np.ndarray,
) -> np.ndarray:
    """Sharpe of every strategy under every partition; shape (n_partitions, n_strategies).

    ``mask`` is (n_partitions, n_splits), holding 1.0 where a block belongs to the set being scored.
    Each aggregation is then a single matrix product, so the whole grid costs three BLAS calls
    rather than a Python loop over partitions.
    """
    n = mask @ counts
    total = mask @ sums
    total_squares = mask @ sums_of_squares

    mean = total / n[:, None]
    # Sample variance from raw moments. Clipped at zero because the subtraction is catastrophic
    # cancellation for a near-constant series and can land a hair below zero.
    variance = np.maximum((total_squares - n[:, None] * mean**2) / (n[:, None] - 1.0), 0.0)
    std = np.sqrt(variance)

    # A strategy with no variation has no Sharpe. Scored as zero rather than infinity so that a
    # degenerate flat series can never win the in-sample selection by dividing through noise. The
    # doubled `where` keeps the division itself off the zero entries instead of dividing and
    # patching up afterwards, which would raise a warning and produce inf en route.
    usable = std > NEGLIGIBLE_STD
    return np.where(usable, mean / np.where(usable, std, 1.0), 0.0)


def _validate_returns_matrix(matrix: np.ndarray, n_splits: int) -> None:
    """Reject shapes and split counts for which the CSCV statistic is undefined or unaffordable."""
    if matrix.ndim != 2:
        raise ValueError(
            f"returns_matrix must be 2-D (observations, strategies), got {matrix.ndim}-D"
        )
    n_observations, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError(
            f"PBO ranks the in-sample winner against its peers, so it needs at least 2 "
            f"strategies; got {n_strategies}"
        )
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even number of at least 2, got {n_splits}")
    if n_observations < n_splits * 2:
        # Each half of a partition must hold enough rows for a variance to mean anything; two
        # observations per block is already the bare minimum.
        raise ValueError(
            f"need at least {n_splits * 2} observations for {n_splits} splits, got {n_observations}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("returns_matrix contains nan or inf; clean or drop them explicitly first")

    n_partitions = math.comb(n_splits, n_splits // 2)
    if n_partitions > MAX_PARTITIONS:
        raise ValueError(
            f"n_splits={n_splits} enumerates {n_partitions:,} partitions, above the "
            f"{MAX_PARTITIONS:,} ceiling; lower n_splits or raise MAX_PARTITIONS deliberately"
        )
    if n_partitions * n_strategies > MAX_GRID_CELLS:
        raise ValueError(
            f"{n_partitions:,} partitions x {n_strategies:,} strategies exceeds the "
            f"{MAX_GRID_CELLS:,} cell working-memory ceiling; reduce n_splits or the corpus size"
        )


def probability_of_backtest_overfitting(returns_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Fraction of train/test partitions in which the in-sample winner underperforms out-of-sample.

    Bailey, Borwein, Lopez de Prado & Zhu (2016), section 3. The series is cut into ``n_splits``
    contiguous blocks; every balanced partition into half training and half testing blocks is
    enumerated; the strategy with the highest in-sample Sharpe is selected and its out-of-sample
    rank recorded. PBO is the share of partitions where that selection lands at or below the
    out-of-sample median.

    Args:
        returns_matrix: shape (n_observations, n_strategies), returns aligned on a common index.
        n_splits: number of contiguous blocks. Must be even. The default of 16 enumerates 12,870
            partitions, the value used in the paper's own experiments.

    Interpretation: 0.5 is what pure noise gives — the in-sample winner is a coin flip
    out-of-sample, so the selection procedure carries no information. Above 0.5 is worse than
    useless. Near 0.0 means the in-sample winner keeps winning, which is what a genuine and
    persistent edge looks like.

    Blocks are contiguous and never shuffled. Shuffling a time series before splitting destroys
    the serial structure the statistic is supposed to be exposed to and would flatter the result.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    _validate_returns_matrix(matrix, n_splits)

    counts, sums, sums_of_squares = _block_moments(matrix, n_splits)
    combinations = np.array(list(itertools.combinations(range(n_splits), n_splits // 2)))
    n_partitions = combinations.shape[0]

    mask = np.zeros((n_partitions, n_splits))
    mask[np.repeat(np.arange(n_partitions), n_splits // 2), combinations.ravel()] = 1.0

    in_sample = _sharpe_grid(mask, counts, sums, sums_of_squares)
    # The complement is the test set, which is what makes the scheme "symmetric": every block
    # serves in training and in testing an equal number of times across the enumeration.
    out_of_sample = _sharpe_grid(1.0 - mask, counts, sums, sums_of_squares)

    selected = np.argmax(in_sample, axis=1)
    selected_oos = out_of_sample[np.arange(n_partitions), selected][:, None]

    # Average rank across ties, so that a cluster of identically-performing strategies neither
    # flatters nor penalises the one that happened to be selected.
    below = (out_of_sample < selected_oos).sum(axis=1)
    tied = (out_of_sample == selected_oos).sum(axis=1)
    rank = below + (tied + 1.0) / 2.0

    relative_rank = rank / (matrix.shape[1] + 1.0)
    # The paper reports PBO as P(logit(w) <= 0) where w is the relative rank. The logit is a
    # monotone transform with logit(w) <= 0 exactly when w <= 0.5, so it is never formed here —
    # computing it would only introduce infinities at the extreme ranks for no change in the answer.
    return float(np.mean(relative_rank <= 0.5))


# --- Tamper-evident trial ledger ------------------------------------------------


class TamperError(LabError):
    """The trial ledger's hash chain does not reconstruct, so its trial count cannot be trusted."""


TrialOutcome = Literal["evaluated", "syntax_error", "runtime_error"]

VALID_OUTCOMES: frozenset[str] = frozenset({"evaluated", "syntax_error", "runtime_error"})

# Fixed genesis string. The first entry chains from its digest, so an attempt to drop the opening
# entries and renumber the rest fails at the very first link.
LEDGER_GENESIS = "trivijaya-quant/trial-ledger/v1"

# Fields that participate in the hash. Order is irrelevant (the payload is serialised with sorted
# keys) but membership is not: anything omitted here can be edited without breaking the chain.
_HASHED_FIELDS = ("seq", "timestamp_utc", "strategy_name", "outcome")


def default_counter_path(cfg: Config) -> Path:
    """Where the trial ledger lives: ``<data_processed>/trial_ledger.jsonl``.

    Deliberately outside the generator's working area. The ledger is the one artefact the strategy
    generator must never be able to write, truncate, or reset — a generator that can edit its own
    trial count can make every deflated Sharpe downstream say whatever it likes. Whatever process
    runs the generator is expected to hold read-only access to this path and to reach the counter
    only through :meth:`TrialCounter.record`.
    """
    return cfg.paths.data_processed / "trial_ledger.jsonl"


def _entry_digest(previous_hash: str, entry: dict[str, Any]) -> str:
    """SHA256 over the previous link's digest plus this entry's canonical payload.

    Serialised with sorted keys and no whitespace so the digest depends on the values alone and not
    on how the JSON happened to be formatted when it was written.
    """
    payload = json.dumps(
        {field: entry[field] for field in _HASHED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256((previous_hash + payload).encode("utf-8")).hexdigest()


class TrialCounter:
    """Append-only, hash-chained ledger of every strategy evaluation attempted.

    One JSON object per line: sequence number, UTC timestamp, strategy name, outcome, the previous
    line's digest, and this line's digest. Each digest covers the previous one, so editing any
    entry breaks every link after it and :meth:`verify` reports where.

    **Every attempt increments the count, including failures and syntax errors.** This is not
    bookkeeping pedantry. The Deflated Sharpe Ratio deflates by N, the number of strategies tried;
    a generator that records only the strategies that ran to completion understates N, which lowers
    the luck threshold, which raises the deflated Sharpe of every survivor. Failed attempts consumed
    search effort exactly as successful ones did and must be counted as trials.

    **What this does and does not protect against.** It detects edits, deletions, reorderings, and
    insertions made by anything that cannot rewrite the whole file — which covers accidental
    corruption and any process holding append-only or no access. It does not defend against an
    actor with full write access who recomputes the entire chain, nor against deletion of the file
    outright. The defence there is access control: see :func:`default_counter_path`.

    Not safe for concurrent writers. Trials in this project are recorded from a single process.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(self, name: str, outcome: TrialOutcome) -> int:
        """Append one trial and return the new total count.

        ``outcome`` has no default on purpose. A caller must state whether the attempt evaluated,
        failed to parse, or blew up at runtime, rather than letting failures slip in as successes.
        """
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}")
        if not name:
            raise ValueError("strategy name must not be empty; an anonymous trial is untraceable")

        entries = self._read_entries()
        previous_hash = entries[-1]["hash"] if entries else self.genesis_hash()
        entry: dict[str, Any] = {
            "seq": len(entries) + 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "strategy_name": name,
            "outcome": outcome,
        }
        entry["prev_hash"] = previous_hash
        entry["hash"] = _entry_digest(previous_hash, entry)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return int(entry["seq"])

    def count(self) -> int:
        """Number of trials recorded. A missing ledger means none have been, and returns 0.

        Counting is not verification. Any N quoted in a report or fed to
        :func:`deflated_sharpe_ratio` must come from a ledger that :meth:`verify` has just passed —
        this method will happily count lines in a file whose chain is broken.
        """
        return len(self._read_lines())

    def verify(self) -> int:
        """Walk the chain and return the number of verified entries. Raises on any break.

        Checks, per entry: the line parses, carries all required fields, has a valid outcome, is
        numbered consecutively from 1, records the previous entry's digest, and hashes to its own
        recorded digest. A missing or empty ledger verifies trivially with a count of 0.
        """
        lines = self._read_lines()
        previous_hash = self.genesis_hash()
        for line_number, line in enumerate(lines, start=1):
            entry = self._parse_entry(line, line_number)
            if entry["seq"] != line_number:
                raise TamperError(
                    f"{self.path} line {line_number}: sequence number is {entry['seq']}, "
                    f"expected {line_number}: an entry was inserted, removed, or reordered"
                )
            if entry["prev_hash"] != previous_hash:
                raise TamperError(
                    f"{self.path} line {line_number}: recorded prev_hash does not match the "
                    f"previous entry's digest, so the chain is broken here"
                )
            recomputed = _entry_digest(previous_hash, entry)
            if recomputed != entry["hash"]:
                raise TamperError(
                    f"{self.path} line {line_number}: contents hash to {recomputed[:16]}... but "
                    f"the line records {str(entry['hash'])[:16]}..., so this entry was edited"
                )
            previous_hash = str(entry["hash"])
        return len(lines)

    @staticmethod
    def genesis_hash() -> str:
        """Digest the first entry chains from, fixing the head of the chain to a known constant."""
        return hashlib.sha256(LEDGER_GENESIS.encode("utf-8")).hexdigest()

    def _parse_entry(self, line: str, line_number: int) -> dict[str, Any]:
        """Decode one ledger line, raising TamperError rather than returning anything partial."""
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TamperError(f"{self.path} line {line_number}: not valid JSON ({exc})") from exc
        if not isinstance(entry, dict):
            raise TamperError(f"{self.path} line {line_number}: expected a JSON object")
        missing = [f for f in (*_HASHED_FIELDS, "prev_hash", "hash") if f not in entry]
        if missing:
            raise TamperError(f"{self.path} line {line_number}: missing fields {missing}")
        if entry["outcome"] not in VALID_OUTCOMES:
            raise TamperError(
                f"{self.path} line {line_number}: unknown outcome {entry['outcome']!r}"
            )
        return dict(entry)

    def _read_lines(self) -> list[str]:
        """Non-blank lines of the ledger, or an empty list if it does not exist yet."""
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]

    def _read_entries(self) -> list[dict[str, Any]]:
        """Parsed entries, used by :meth:`record` to find the tail of the chain."""
        return [self._parse_entry(line, i) for i, line in enumerate(self._read_lines(), start=1)]
