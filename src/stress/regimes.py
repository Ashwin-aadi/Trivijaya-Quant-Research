"""Expanding-window HMM market-regime labelling in which every label is causal at its session.

Fitting on past data only is necessary but *not sufficient*, and that distinction is the whole
design of this module. A Gaussian HMM fit on sessions strictly before ``t`` can still leak at the
decoding step, because the conventional decoders look forward:

* ``hmmlearn.GaussianHMM.predict`` is Viterbi over the supplied sequence. It picks the state at
  ``t`` partly from observations *after* ``t``, since the most likely path depends on where the
  path ends.
* ``hmmlearn.GaussianHMM.predict_proba`` returns smoothed posteriors ``P(state_t | obs_1..T)``,
  which condition on the entire sequence by construction.

Both are forbidden here (PI ruling, DECISIONS.md Phase 2.0 decision 1). A label is instead the
**filtered** estimate

    argmax_k  P(state_t = k | obs_1 .. obs_t)

computed by the forward recursion alone. Only the fitting half of ``hmmlearn`` is used; the decode
is implemented here so the causal claim is auditable in one place rather than resting on a library
call whose semantics could change.

The consequence is stated rather than hidden: filtered labels are noisier than smoothed ones and
lag turning points. That lag is what an observer standing at session ``t`` could actually have
known. A label that identifies the top of the market on the day it happens is using tomorrow's
information.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from src.common.exceptions import LabError
from src.common.log import get_logger

_log = get_logger(__name__)

#: The frozen feature set (PI, DECISIONS.md Phase 2.0 decision 4). Order is load-bearing:
#: canonicalisation sorts states on feature 0, so volatility must come first.
FEATURE_NAMES: tuple[str, ...] = ("log_realised_vol", "cum_log_return")

#: One trading month. Reused from ``constraints.adv_window_sessions`` in config.yaml rather than
#: introduced as a new free parameter.
FEATURE_WINDOW = 21

#: Sessions per year, for annualising volatility. Presentational only — a constant scale factor
#: cannot change which state a session is assigned to.
ANNUALISATION = 252

#: Floor on training rows before any label is emitted. Two years of daily observations to estimate
#: a K-state Gaussian HMM on 2 features: at K=3 that is 24 free parameters, so ~21 observations
#: each. Under the approved burn-in the first refit already has ~1,200 rows, so this floor never
#: binds in practice — it exists so that a future run on a shorter history fails loudly instead of
#: quietly fitting a model on nothing.
MIN_TRAIN_SESSIONS = 504


class RegimeError(LabError):
    """A regime model was fit, decoded, or labelled outside the conditions it is valid under."""


# --- features -------------------------------------------------------------------


def build_features(
    frame: pl.DataFrame,
    window: int = FEATURE_WINDOW,
    date_column: str = "session_date",
    close_column: str = "close",
) -> pl.DataFrame:
    """Build the two frozen regime features, each using only sessions up to and including ``t``.

    Returns one row per session that has a full trailing window, so the warm-up rows are dropped
    rather than back-filled — an imputed early feature would be a value no observer ever saw.

    Volatility is logged because it is approximately lognormal, and a Gaussian emission fit to raw
    volatility is badly mis-specified: it would put appreciable mass on negative volatility.
    """
    if window < 2:
        raise RegimeError(f"feature window must be at least 2 sessions, got {window}")
    ordered = frame.select(date_column, close_column).sort(date_column)
    if ordered.height < window + 1:
        raise RegimeError(f"need at least {window + 1} sessions to build features")

    log_return = (pl.col(close_column).log() - pl.col(close_column).shift(1).log()).alias("r")
    # `rolling_*` over a window ending at the current row, and no negative shifts anywhere: every
    # value at row t is a function of rows <= t only. This is the point-in-time guarantee.
    built = (
        ordered.with_columns(log_return)
        .with_columns(
            (pl.col("r").rolling_std(window, ddof=1) * np.sqrt(ANNUALISATION))
            .log()
            .alias(FEATURE_NAMES[0]),
            pl.col("r").rolling_sum(window).alias(FEATURE_NAMES[1]),
        )
        .drop_nulls(list(FEATURE_NAMES))
        .select(date_column, *FEATURE_NAMES)
    )
    if built.height == 0:
        raise RegimeError("feature construction produced no rows")
    return built


def feature_matrix(features: pl.DataFrame) -> np.ndarray:
    """The feature columns as a float64 matrix, in the frozen order."""
    return features.select(*FEATURE_NAMES).to_numpy().astype(np.float64)


# --- the model ------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeModel:
    """A fitted, canonicalised Gaussian HMM plus the provenance needed to audit its causality.

    ``fitted_through`` is the last session that entered the fit. Every label this model produces
    must be for a session strictly after it, which :func:`expanding_window_labels` enforces.
    """

    k: int
    startprob: np.ndarray      # (K,)
    transmat: np.ndarray       # (K, K)
    means: np.ndarray          # (K, D)
    covars: np.ndarray         # (K, D, D)
    n_train: int
    fitted_through: date
    log_likelihood: float
    #: Restarts that failed to converge to a valid covariance. Carried so the count reaches
    #: the diagnostics rather than living only in a log line nobody reads.
    failed_restarts: int = 0

    def emission_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """Per-state Gaussian log-density of each row of ``x``. Shape (T, K)."""
        return _gaussian_log_likelihood(x, self.means, self.covars)

    def filtered_log_posteriors(self, x: np.ndarray) -> np.ndarray:
        """log P(state_t | obs_1..t) for every row. Forward recursion only — never backward."""
        return self._filter(x)[0]

    def filtered_states(self, x: np.ndarray) -> np.ndarray:
        """The causal label for each row: the most probable state given observations up to it."""
        return np.asarray(np.argmax(self.filtered_log_posteriors(x), axis=1), dtype=np.int64)

    def sequence_log_evidence(self, x: np.ndarray) -> float:
        """log P(obs_1..T) under this model, accumulated from the same forward pass.

        Exists so the hand-written recursion can be checked against ``hmmlearn``'s own ``score``
        on identical parameters — see ``test_forward_recursion_matches_hmmlearn_evidence``. Without
        that cross-check the decode would be trusted on inspection alone.
        """
        return self._filter(x)[1]

    def _filter(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        return forward_filter(
            _safe_log(self.startprob), _safe_log(self.transmat), self.emission_log_likelihood(x)
        )


def _gaussian_log_likelihood(x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """Multivariate normal log-density of each observation under each state.

    Written out rather than taken from hmmlearn's private ``_compute_log_likelihood`` so that the
    decode path depends on no private API, and so a reader can check it against the standard form
    without leaving this file.
    """
    n_obs, dim = x.shape
    out = np.empty((n_obs, means.shape[0]), dtype=np.float64)
    for state in range(means.shape[0]):
        cov = covars[state]
        # Cholesky rather than an explicit inverse: numerically stabler, and it raises on a
        # non-positive-definite covariance instead of returning quiet nonsense.
        chol = np.linalg.cholesky(cov)
        deviation = x - means[state]
        solved = _solve_lower(chol, deviation.T)
        mahalanobis = np.sum(solved**2, axis=0)
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        out[:, state] = -0.5 * (mahalanobis + log_det + dim * np.log(2.0 * np.pi))
    return out


def _solve_lower(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``lower @ z = rhs`` for a lower-triangular ``lower`` by forward substitution."""
    from scipy.linalg import solve_triangular

    return np.asarray(solve_triangular(lower, rhs, lower=True), dtype=np.float64)


def forward_filter(
    log_startprob: np.ndarray,
    log_transmat: np.ndarray,
    frame_log_likelihood: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Normalised forward recursion in log space.

    Returns ``(log P(state_t | obs_1..t) of shape (T, K), log P(obs_1..T))``.

    Each row is normalised as it is produced, which is what makes the result a *filtered* posterior
    rather than an unnormalised forward variable, and keeps the recursion numerically bounded over
    sequences of thousands of sessions. The discarded normalisers are exactly the one-step-ahead
    predictive likelihoods, so accumulating them gives the sequence evidence for free.

    Nothing in this function reads ``frame_log_likelihood`` beyond row ``t`` when computing row
    ``t``. That property is the whole causality guarantee and is asserted by
    ``test_filtered_label_ignores_the_future``.
    """
    n_obs, n_states = frame_log_likelihood.shape
    out = np.empty((n_obs, n_states), dtype=np.float64)
    current = log_startprob + frame_log_likelihood[0]
    normaliser = _logsumexp(current)
    evidence = float(normaliser)
    out[0] = current - normaliser
    for t in range(1, n_obs):
        # log sum_j P(state_{t-1}=j | obs_1..t-1) * A[j, k], then multiply in the emission at t.
        predicted = _logsumexp(out[t - 1][:, None] + log_transmat, axis=0)
        current = predicted + frame_log_likelihood[t]
        normaliser = _logsumexp(current)
        evidence += float(normaliser)
        out[t] = current - normaliser
    return out, evidence


def _safe_log(values: np.ndarray) -> np.ndarray:
    """``log`` mapping exact zeros to ``-inf`` without a warning.

    A transition or start probability of exactly zero is legitimate — hmmlearn produces them when a
    state is unreachable — and ``-inf`` is the correct log. Suppressed here so a genuine numerical
    problem elsewhere is not lost among expected warnings.
    """
    with np.errstate(divide="ignore"):
        return np.asarray(np.log(values), dtype=np.float64)


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp, tolerating all-``-inf`` inputs."""
    peak = np.max(values, axis=axis, keepdims=True)
    # An all -inf slice has peak -inf; shifting by it would give nan, so shift by 0 and let the
    # sum underflow to 0 and the log return -inf, which is the right answer.
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    return np.asarray(np.squeeze(total, axis=axis) if axis is not None else total.reshape(()))


# --- fitting and canonicalisation -----------------------------------------------


def canonical_permutation(means: np.ndarray) -> np.ndarray:
    """State ordering: ascending by emission mean of feature 0 (log volatility), so 0 is calmest.

    HMM state indices are arbitrary and are re-drawn at every refit — state 0 in one quarter's fit
    need not be state 0 in the next. Without this, the stability diagnostic would measure label
    permutation noise instead of regime change, and would report chaos regardless of the truth
    (PI, DECISIONS.md Phase 2.0 decision 2).
    """
    return np.asarray(np.argsort(means[:, 0], kind="stable"), dtype=np.int64)


#: Random restarts per fit. Baum-Welch is EM: it finds a local optimum, and which one depends on
#: the initialisation. Measured on the burn-in window at K=3, a single start landed on a degenerate
#: solution with two duplicate states and a log-likelihood BELOW the K=2 fit — impossible for a
#: correctly converged nested family, and enough to corrupt a BIC comparison against it.
#:
#: Restarts are a convergence fix, NOT a tuning knob: the criterion, the candidate state counts and
#: the fitting window are all unchanged, and the selection among restarts is on training likelihood
#: alone. No backtest, no performance metric. See DECISIONS.md, Phase 2.0 convergence note.
N_RESTARTS = 8


def fit_regime_model(
    x: np.ndarray,
    k: int,
    fitted_through: date,
    seed: int,
    n_iter: int = 200,
    n_restarts: int = N_RESTARTS,
) -> RegimeModel:
    """Fit a K-state full-covariance Gaussian HMM and return it in canonical state order.

    Fits ``n_restarts`` times from deterministically-derived seeds and keeps the highest training
    log-likelihood, so a single unlucky initialisation cannot masquerade as evidence that K states
    fit the data poorly.

    ``x`` must contain only observations the caller is permitted to have seen; this function has no
    way to check that and does not try. :func:`expanding_window_labels` is what enforces it.
    """
    from hmmlearn.hmm import GaussianHMM

    if x.shape[0] < MIN_TRAIN_SESSIONS:
        raise RegimeError(
            f"refusing to fit on {x.shape[0]} sessions; the floor is {MIN_TRAIN_SESSIONS}"
        )
    if n_restarts < 1:
        raise RegimeError(f"n_restarts must be at least 1, got {n_restarts}")

    model = None
    best_score = -np.inf
    failures: list[str] = []
    for restart in range(n_restarts):
        candidate = GaussianHMM(
            n_components=k,
            covariance_type="full",
            n_iter=n_iter,
            # Derived from the global seed, so the whole restart sweep is reproducible
            # (charter RULE 6: no unseeded randomness anywhere).
            random_state=seed + 1000 * restart,
            tol=1e-4,
        )
        try:
            candidate.fit(x)
            score = float(candidate.score(x))
        except (ValueError, np.linalg.LinAlgError) as exc:
            # A restart can drive a state's covariance to be non-positive-definite — a state
            # collapsing onto near-identical points. That is a failed *initialisation*, not a
            # failed model, and it is exactly what the other restarts exist to survive.
            #
            # This is NOT a swallowed exception: the failure is logged at WARNING, counted, carried
            # on the returned model, and re-raised if every restart fails. A silent `except: pass`
            # here would hide a genuinely unfittable K.
            failures.append(f"restart {restart}: {type(exc).__name__}: {exc}")
            _log.warning("K=%d restart %d failed: %s", k, restart, exc)
            continue
        if score > best_score:
            best_score, model = score, candidate

    if model is None:
        raise RegimeError(
            f"all {n_restarts} restarts failed to fit K={k} on {x.shape[0]} sessions:\n"
            + "\n".join(failures)
        )

    order = canonical_permutation(model.means_)
    start = np.asarray(model.startprob_, dtype=np.float64)[order]
    return RegimeModel(
        k=k,
        # Renormalised after permuting: the reordered vector is a permutation of a simplex point,
        # so this is a no-op mathematically and a guard against accumulated float drift.
        startprob=start / start.sum(),
        transmat=np.asarray(model.transmat_, dtype=np.float64)[np.ix_(order, order)],
        means=np.asarray(model.means_, dtype=np.float64)[order],
        covars=np.asarray(model.covars_, dtype=np.float64)[order],
        n_train=int(x.shape[0]),
        fitted_through=fitted_through,
        log_likelihood=float(model.score(x)),
        failed_restarts=len(failures),
    )


def n_free_parameters(k: int, dim: int) -> int:
    """Free parameters of a K-state full-covariance Gaussian HMM on ``dim`` features.

    startprob (K-1, simplex) + transmat (K rows each on a simplex, K(K-1)) + means (K*dim)
    + full covariances (K * dim(dim+1)/2, symmetric).
    """
    return (k - 1) + k * (k - 1) + k * dim + k * dim * (dim + 1) // 2


def bayesian_information_criterion(model: RegimeModel, x: np.ndarray) -> float:
    """BIC = -2 log L + p log n. Lower is better.

    Used to select K once, on the burn-in window only, after which K is frozen permanently
    (PI, DECISIONS.md Phase 2.0 decision 5). BIC is a likelihood-penalty criterion: no strategy is
    run and no performance metric is consulted, so selecting on it is not the post-hoc parameter
    search that P2's design rules out.
    """
    n_obs, dim = x.shape
    return -2.0 * model.log_likelihood + n_free_parameters(model.k, dim) * float(np.log(n_obs))


# --- the expanding-window driver ------------------------------------------------


def expanding_window_labels(
    features: pl.DataFrame,
    refit_dates: list[date],
    k: int,
    seed: int,
    min_train: int = MIN_TRAIN_SESSIONS,
    date_column: str = "session_date",
) -> tuple[pl.DataFrame, list[RegimeModel]]:
    """Label every session using only a model fit before it and observations up to it.

    At each refit date the model is re-estimated on all feature rows *strictly before* that date,
    then used to label the sessions from that date until the next refit. Both halves of the causal
    guarantee are enforced here rather than assumed:

    * **Parameters** come from ``sessions < refit_date``.
    * **Observations** entering the filter at session ``t`` are ``<= t``, by construction of
      :func:`forward_filter`.

    Sessions before the first refit that clears ``min_train`` receive no label at all. They are
    absent from the output rather than carrying a null, so a downstream join cannot silently treat
    an unlabelled session as a labelled one.

    Returns the labels and the fitted models, one per refit that produced any.
    """
    dates = features[date_column].to_list()
    matrix = feature_matrix(features)
    boundaries = sorted(refit_dates)
    labelled: list[pl.DataFrame] = []
    models: list[RegimeModel] = []

    for position, refit_date in enumerate(boundaries):
        train_rows = sum(1 for day in dates if day < refit_date)
        if train_rows < min_train:
            continue
        next_boundary = boundaries[position + 1] if position + 1 < len(boundaries) else None
        segment = [
            i
            for i, day in enumerate(dates)
            if day >= refit_date and (next_boundary is None or day < next_boundary)
        ]
        if not segment:
            continue
        model = fit_regime_model(
            matrix[:train_rows], k=k, fitted_through=dates[train_rows - 1], seed=seed
        )
        models.append(model)
        # Filter over everything up to the end of this segment, then keep only the segment's rows.
        # Row i of the recursion conditions on rows <= i, so taking row i is causal at date[i].
        states = model.filtered_states(matrix[: segment[-1] + 1])
        labelled.append(
            pl.DataFrame(
                {
                    date_column: [dates[i] for i in segment],
                    "state": [int(states[i]) for i in segment],
                    "refit_date": [refit_date] * len(segment),
                    "n_train": [model.n_train] * len(segment),
                }
            )
        )

    if not labelled:
        raise RegimeError(
            f"no refit date had {min_train} training sessions before it; nothing was labelled"
        )
    return pl.concat(labelled).sort(date_column), models
