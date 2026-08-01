"""Counterfactual Regime Resampling: synthetic-but-plausible market histories.

History gave one path. Indian markets since 2015 contain approximately one pandemic crash, one
full rate cycle and one structural liquidity shift, so the number of *independent regime
observations* is very small — testing a strategy against the regimes that happened to occur is
weak evidence about how it behaves under regimes that did not. CRR addresses that scarcity by
resampling the observed history in blocks, preserving the dependence structure that makes a path
plausible rather than drawing days independently.

**What this module resamples: dates, not values.** Every function here returns *integer index
sequences* into the original observation order. Those indices are then applied identically to
every column of whatever panel the caller holds, which is what preserves the cross-sectional
correlation structure — on any synthetic day, the co-movement between two series is a real
co-movement that actually occurred on some real day. Resampling series independently would destroy
exactly the structure a portfolio's risk depends on.

That design also keeps this module agnostic about *what* is being resampled, which is deliberate:
the choice of resampling target is a methodological fork for the PI, not an assumption to bury in
a library.

References, cited because the block-length rule is not a free parameter here:

* Politis, D.N. and Romano, J.P. (1994). "The Stationary Bootstrap."
  *Journal of the American Statistical Association* 89(428), 1303-1313.
* Politis, D.N. and White, H. (2004). "Automatic Block-Length Selection for the Dependent
  Bootstrap." *Econometric Reviews* 23(1), 53-70.
* Patton, A., Politis, D.N. and White, H. (2009). "Correction to 'Automatic Block-Length Selection
  for the Dependent Bootstrap'." *Econometric Reviews* 28(4), 372-375.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.exceptions import LabError
from src.common.log import get_logger

_log = get_logger(__name__)


class ResamplingError(LabError):
    """A resampling was requested under conditions that would make the output meaningless."""


# --- the stationary bootstrap ---------------------------------------------------


def stationary_bootstrap_indices(
    n_obs: int,
    block_length: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Index paths from the stationary bootstrap of Politis & Romano (1994).

    Returns an ``(n_paths, n_obs)`` integer array. Each row is a sequence of positions into the
    original observations; applying one row to every column of a panel yields one synthetic history
    with the cross-section intact.

    The procedure: start at a uniformly random position. At each subsequent step, with probability
    ``p = 1 / block_length`` jump to a new uniformly random position, otherwise advance one step
    from the current position, wrapping at the end. Block lengths are therefore geometric with mean
    ``block_length``, and the resulting series is strictly stationary — which is the property that
    distinguishes this from the fixed-block bootstrap and the reason it is the standard choice for
    financial return series.

    Wrapping joins the end of the sample to its beginning. That creates a small number of synthetic
    transitions between periods that were never adjacent — an acknowledged artefact of the method,
    material only if the series has a strong trend in level. Returns do not; prices would, which is
    a reason to resample returns rather than levels.
    """
    if n_obs < 2:
        raise ResamplingError(f"need at least 2 observations to resample, got {n_obs}")
    if not np.isfinite(block_length) or block_length < 1.0:
        raise ResamplingError(f"block_length must be finite and at least 1, got {block_length}")
    if n_paths < 1:
        raise ResamplingError(f"n_paths must be at least 1, got {n_paths}")

    rng = np.random.default_rng(seed)   # charter RULE 6: explicit seed, no global state
    jump_probability = 1.0 / block_length

    out = np.empty((n_paths, n_obs), dtype=np.int64)
    starts = rng.integers(0, n_obs, size=(n_paths,))
    # Draw all randomness up front in two arrays rather than per-step: identical distribution,
    # and it keeps a 1000 x 1233 draw in the sub-second range instead of the tens of seconds a
    # Python-level per-step loop costs.
    jumps = rng.random((n_paths, n_obs)) < jump_probability
    fresh = rng.integers(0, n_obs, size=(n_paths, n_obs))

    out[:, 0] = starts
    for step in range(1, n_obs):
        advanced = (out[:, step - 1] + 1) % n_obs
        out[:, step] = np.where(jumps[:, step], fresh[:, step], advanced)
    return out


# --- automatic block-length selection -------------------------------------------


def _flat_top_kernel(values: np.ndarray) -> np.ndarray:
    """Trapezoidal flat-top lag window of Politis & White (2004), their equation (7)."""
    absolute = np.abs(values)
    return np.where(absolute <= 0.5, 1.0, np.where(absolute <= 1.0, 2.0 * (1.0 - absolute), 0.0))


def _autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample autocovariances at lags 0..max_lag (divisor n, as Politis & White assume)."""
    n_obs = series.shape[0]
    centred = series - series.mean()
    return np.array(
        [float(np.dot(centred[: n_obs - lag], centred[lag:]) / n_obs) for lag in range(max_lag + 1)]
    )


def _characteristic_lag(series: np.ndarray) -> int:
    """The smallest lag beyond which the correlogram is negligible.

    Politis & White's ``m_hat``: the smallest ``m`` such that every one of the next ``K_N``
    autocorrelations falls inside the band ``2 sqrt(log10(n) / n)``. The band is the implied
    significance threshold for a white-noise null; ``M = 2 m_hat`` is then used as the bandwidth.
    """
    n_obs = series.shape[0]
    max_lag = max(5, int(np.ceil(np.sqrt(np.log10(n_obs)))))
    search_limit = min(n_obs - 1, int(np.ceil(5.0 * np.sqrt(np.log10(n_obs)))) + max_lag + 20)
    acov = _autocovariance(series, search_limit)
    if acov[0] <= 0.0:
        raise ResamplingError("series has zero variance; block length is undefined")
    acf = acov / acov[0]
    threshold = 2.0 * np.sqrt(np.log10(n_obs) / n_obs)

    for candidate in range(1, search_limit - max_lag + 1):
        window = acf[candidate + 1 : candidate + 1 + max_lag]
        if window.size and np.all(np.abs(window) < threshold):
            return int(candidate)
    # Nothing satisfied the criterion: the series stays correlated as far as we looked. Returning
    # the search limit rather than 1 keeps the resulting block length long, which is the
    # conservative direction — too short a block destroys the dependence the method exists to keep.
    _log.warning("no lag satisfied the Politis-White criterion; using the search limit")
    return int(max(1, search_limit - max_lag))


@dataclass(frozen=True)
class BlockLengthEstimate:
    """A block length with the intermediate quantities that produced it, for auditability."""

    block_length: float
    characteristic_lag: int
    bandwidth: int
    g_hat: float
    d_hat: float
    n_obs: int


def optimal_block_length(series: np.ndarray) -> BlockLengthEstimate:
    """Automatic block length for the stationary bootstrap, Politis & White (2004) §3.

    ``b_opt = (2 G^2 / D_SB)^(1/3) n^(1/3)``, with ``G = sum_k lambda(k/M) |k| R(k)`` and
    ``D_SB = 2 g(0)^2`` where ``g(0) = sum_k lambda(k/M) R(k)``, both summed over
    ``k = -M..M`` and using the flat-top kernel. The ``D_SB`` form is the stationary-bootstrap
    case as corrected in Patton, Politis & White (2009); the circular-bootstrap constant differs
    and using it here would bias the block length.

    The result is clipped to ``[1, n^(1/3) * 3]``, the upper bound recommended in the same paper,
    so a near-unit-root series cannot produce a block length approaching the sample size — at which
    point the "bootstrap" would be returning the original path with a rotation.
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    n_obs = series.shape[0]
    if n_obs < 16:
        raise ResamplingError(
            f"need at least 16 observations to estimate a block length, got {n_obs}"
        )

    characteristic = _characteristic_lag(series)
    bandwidth = max(1, 2 * characteristic)
    acov = _autocovariance(series, min(bandwidth, n_obs - 1))
    lags = np.arange(acov.shape[0])
    weights = _flat_top_kernel(lags / bandwidth)

    # Sums run over k = -M..M and the autocovariance is symmetric, so double the positive lags and
    # add lag 0 once. Writing it this way keeps it checkable against the paper's notation.
    g_hat = float(2.0 * np.sum(weights[1:] * lags[1:] * acov[1:]))
    g_zero = float(acov[0] + 2.0 * np.sum(weights[1:] * acov[1:]))
    d_hat = 2.0 * g_zero**2

    if d_hat <= 0.0 or g_hat == 0.0:
        # A series with no usable dependence structure: fall back to a single-observation block,
        # which reduces the stationary bootstrap to the iid bootstrap. Stated, not silent.
        _log.warning(
            "degenerate block-length estimate (G=%g, D=%g); falling back to 1", g_hat, d_hat
        )
        return BlockLengthEstimate(1.0, characteristic, bandwidth, g_hat, d_hat, n_obs)

    raw = float((2.0 * g_hat**2 / d_hat) ** (1.0 / 3.0) * n_obs ** (1.0 / 3.0))
    upper = 3.0 * n_obs ** (1.0 / 3.0)
    return BlockLengthEstimate(
        block_length=float(np.clip(raw, 1.0, upper)),
        characteristic_lag=characteristic,
        bandwidth=bandwidth,
        g_hat=g_hat,
        d_hat=d_hat,
        n_obs=n_obs,
    )


# --- conditioning ---------------------------------------------------------------


def conditional_bootstrap_indices(
    labels: np.ndarray,
    block_length: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Stationary bootstrap that may only jump to a position sharing the current label.

    Unconditional resampling can splice a calm stretch directly onto a crisis, producing paths
    whose regime composition never occurred. Restricting jumps to positions with the same label
    keeps a synthetic path's local character intact: within a block the sequence is real and
    contiguous, and at a jump the destination is at least the same *kind* of market.

    Note honestly what this does and does not preserve. It preserves the marginal distribution of
    each label's returns and the within-block dynamics. It does **not** preserve the historical
    *transition* structure between regimes — a path can dwell in one label far longer than history
    ever did, because jumps never change label. Regime persistence in the synthetic paths is
    therefore an artefact of the block length, not an estimate of anything.
    """
    labels = np.asarray(labels).ravel()
    n_obs = labels.shape[0]
    if n_obs < 2:
        raise ResamplingError(f"need at least 2 observations to resample, got {n_obs}")
    if not np.isfinite(block_length) or block_length < 1.0:
        raise ResamplingError(f"block_length must be finite and at least 1, got {block_length}")

    positions_by_label = {
        int(label): np.flatnonzero(labels == label) for label in np.unique(labels)
    }
    thin = {label: pos for label, pos in positions_by_label.items() if pos.size < 2}
    if thin:
        raise ResamplingError(
            f"labels {sorted(thin)} have fewer than 2 observations; conditioning on them would "
            "resample a single day repeatedly rather than bootstrap it"
        )

    rng = np.random.default_rng(seed)
    jump_probability = 1.0 / block_length
    out = np.empty((n_paths, n_obs), dtype=np.int64)
    jumps = rng.random((n_paths, n_obs)) < jump_probability

    for path in range(n_paths):
        current = int(rng.integers(0, n_obs))
        out[path, 0] = current
        for step in range(1, n_obs):
            if jumps[path, step]:
                pool = positions_by_label[int(labels[current])]
                current = int(pool[rng.integers(0, pool.size)])
            else:
                current = (current + 1) % n_obs
            out[path, step] = current
    return out
