"""Tier 2: fragility from resampled realised returns, without re-running any strategy.

Tier 1 rebuilds the price panel and lets each strategy *re-decide* under a counterfactual history.
That is faithful and it costs five and a half hours. Tier 2 takes the return series a strategy
actually realised and block-bootstraps it directly, which costs minutes. The difference is not
cosmetic: a resampled return series holds the strategy's *decisions* fixed and reshuffles only
their outcomes, so it cannot capture a strategy that would have positioned differently. Tier 2 is
therefore an approximation, and the point of running both tiers over the same strategies is that
the size of that approximation is **measured rather than assumed**.

Two fragility definitions are computed, because they are not the same quantity and the project
charter and the available Tier 1 output disagree about which one is meant.

``across_regimes``
    ``F = Var_r[pi_r] / |E_r[pi_r]|`` over the regime labels of the *sampled* sessions. This is the
    charter's definition. It asks: does this strategy perform consistently across kinds of market?

``across_paths``
    ``F = Var_p[pi_p] / |E_p[pi_p]|`` over whole synthetic paths. This asks: how much does the
    strategy's overall result depend on which counterfactual history it met? It is the only one
    Tier 1's stored output can supply, because Tier 1 kept per-path summaries and not daily returns.

Both are ratios with a mean in the denominator, and a strategy whose mean performance is near zero
will show an enormous fragility for an arithmetic reason rather than a behavioural one. That is a
property of the definition, not a bug, and it is why ``mean_is_near_zero`` is reported alongside
every value instead of being silently smoothed away.

Bootstrap machinery is imported from :mod:`src.stress.crr` rather than reimplemented, so Tier 1 and
Tier 2 resample by the identical rule (Politis & Romano 1994; block length by Politis & White 2004).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.stress.crr import conditional_bootstrap_indices, stationary_bootstrap_indices

#: Trading sessions per year on the NSE calendar, for annualising. Matches src.eval.metrics.
SESSIONS_PER_YEAR = 252.0

#: Below this absolute mean, a fragility ratio is dominated by its denominator. Strategies under it
#: are reported with a flag rather than dropped: a near-zero mean is itself informative, and hiding
#: the resulting large F would misrepresent how many strategies the metric cannot rank.
NEAR_ZERO_MEAN = 0.05

#: Volatility at or below this is treated as zero. See :func:`_sharpe` for why ``std > 0`` is not
#: a sufficient test and what it produced when it was used.
DEGENERATE_VOLATILITY = 1e-10

#: A regime slice thinner than this cannot support a Sharpe ratio worth reporting. Roughly one
#: trading month; below it the volatility estimate in the denominator is too noisy to interpret.
MIN_REGIME_SESSIONS = 21


@dataclass(frozen=True)
class FragilityResult:
    """Every fragility number for one strategy under one variant, with its sample sizes."""

    name: str
    variant: str
    n_paths: int
    n_sessions: int
    mean_path_sharpe: float
    std_path_sharpe: float
    fragility_across_paths: float
    fragility_across_regimes: float
    #: Mean Sharpe within each regime label, averaged over paths. Index is the label value.
    regime_sharpe: dict[int, float] = field(default_factory=dict)
    #: Sessions per regime in the real series — the sample size behind each regime_sharpe entry.
    regime_sessions: dict[int, int] = field(default_factory=dict)
    mean_is_near_zero: bool = False
    knife_edge: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "variant": self.variant,
            "n_paths": self.n_paths,
            "n_sessions": self.n_sessions,
            "mean_path_sharpe": self.mean_path_sharpe,
            "std_path_sharpe": self.std_path_sharpe,
            "fragility_across_paths": self.fragility_across_paths,
            "fragility_across_regimes": self.fragility_across_regimes,
            "regime_sharpe": {str(k): v for k, v in self.regime_sharpe.items()},
            "regime_sessions": {str(k): v for k, v in self.regime_sessions.items()},
            "mean_is_near_zero": self.mean_is_near_zero,
            "knife_edge": self.knife_edge,
        }


def draw_paths(
    labels: np.ndarray, block_length: float, n_paths: int, seed: int, *, conditional: bool
) -> np.ndarray:
    """One shared ``(n_paths, n_sessions)`` index matrix, used by every strategy of this length.

    Sharing matters. If each strategy drew its own paths, two strategies could differ because they
    met different counterfactual histories rather than because they respond differently, and the
    cross-sectional comparison that Phase 2.2 trains on would be partly resampling noise.
    """
    if conditional:
        return conditional_bootstrap_indices(labels, block_length, n_paths, seed)
    return stationary_bootstrap_indices(labels.shape[0], block_length, n_paths, seed)


def _sharpe(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    """Annualised Sharpe along an axis, vectorised over paths. Degenerate volatility yields zero.

    The guard is an absolute floor, not ``std > 0``. A series of identical values does not have a
    standard deviation of exactly zero in floating point — ``np.std(ddof=1)`` over 79 copies of
    0.001 returns 2.2e-19, from the residuals of the mean subtraction. Testing ``std > 0`` therefore
    passes, and the ratio comes back as 7e16 rather than 0. That is not hypothetical here: a
    strategy holding cash through a regime has exactly-constant returns over that slice, and would
    have contributed an astronomically large Sharpe to its regime fragility.

    ``1e-10`` sits far below any real daily-return volatility — the least volatile strategy in the
    corpus realises about 4e-4 — and far above the 1e-19 scale of the arithmetic residual.
    """
    mean = returns.mean(axis=axis)
    std = returns.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(std > DEGENERATE_VOLATILITY, mean / std, 0.0)
    return np.asarray(ratio * np.sqrt(SESSIONS_PER_YEAR))


def _ratio(values: np.ndarray) -> tuple[float, bool]:
    """``Var / |mean|`` with the near-zero-denominator case flagged rather than hidden."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan"), True
    mean = float(np.mean(finite))
    variance = float(np.var(finite, ddof=1))
    if abs(mean) < NEAR_ZERO_MEAN:
        return variance / max(abs(mean), 1e-12), True
    return variance / abs(mean), False


def fragility(
    name: str,
    returns: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    *,
    variant: str,
    knife_edge: bool = False,
) -> FragilityResult:
    """Both fragility definitions for one strategy, over a pre-drawn set of index paths.

    ``paths`` carries source positions, so each resampled session brings its own regime label with
    it. That is what makes the across-regime figure meaningful under resampling: a synthetic path's
    regime composition is whatever the bootstrap drew, not the calendar's.
    """
    sampled = returns[paths]                     # (n_paths, n_sessions)
    sampled_labels = labels[paths]
    path_sharpe = _sharpe(sampled, axis=1)
    across_paths, paths_near_zero = _ratio(path_sharpe)

    # Per regime, per path, then averaged over paths. Regimes too thin in the real series are
    # excluded outright rather than contributing a Sharpe built on a handful of sessions.
    per_regime: dict[int, float] = {}
    sessions: dict[int, int] = {}
    for label in np.unique(labels):
        real_count = int(np.sum(labels == label))
        if real_count < MIN_REGIME_SESSIONS:
            continue
        mask = sampled_labels == label
        # Ragged across paths: each path draws a different number of sessions per regime, so the
        # Sharpe is computed path by path and only then averaged.
        values = [
            _sharpe(sampled[p][mask[p]])
            for p in range(sampled.shape[0])
            if int(mask[p].sum()) >= MIN_REGIME_SESSIONS
        ]
        if not values:
            continue
        per_regime[int(label)] = float(np.mean(values))
        sessions[int(label)] = real_count

    across_regimes, regimes_near_zero = _ratio(np.array(list(per_regime.values())))

    return FragilityResult(
        name=name,
        variant=variant,
        n_paths=int(paths.shape[0]),
        n_sessions=int(returns.shape[0]),
        mean_path_sharpe=float(np.mean(path_sharpe)),
        std_path_sharpe=float(np.std(path_sharpe, ddof=1)),
        fragility_across_paths=across_paths,
        fragility_across_regimes=across_regimes,
        regime_sharpe=per_regime,
        regime_sessions=sessions,
        mean_is_near_zero=paths_near_zero or regimes_near_zero,
        knife_edge=knife_edge,
    )
