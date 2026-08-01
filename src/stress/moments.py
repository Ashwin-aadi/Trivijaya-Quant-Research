"""Compare synthetic resampled paths against the real series they were drawn from.

The charter is explicit that synthetic sequences must be *shown* to reproduce the empirical
distribution of key moments, not asserted to. This module computes the comparison and reports
where it fails, because the places a bootstrap breaks down are more informative than the places it
succeeds — and a validation that only ever reports success is not a validation.

The statistics chosen are the ones a block bootstrap can plausibly get wrong:

* **volatility clustering** — autocorrelation of absolute returns. This is the property the whole
  method exists to preserve; if it collapses, the block length is too short.
* **return autocorrelation** — should be near zero in both, and stay near zero.
* **skewness and kurtosis** — fat tails and asymmetry survive resampling only if extreme days are
  drawn in their original neighbourhoods.
* **maximum drawdown** — a path-dependent statistic no moment-matching argument covers. It is the
  one most likely to be wrong, and the one that matters most for a stress test.

Each statistic is reported as the real value against the synthetic distribution, with the
**empirical percentile of the real value within that distribution**. A percentile near 0 or 100
means the synthetic paths systematically fail to reproduce reality on that axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.exceptions import LabError


class MomentError(LabError):
    """A moment comparison was requested on data that cannot support it."""


@dataclass(frozen=True)
class MomentComparison:
    """One statistic: its real value, the synthetic distribution, and where the real value sits."""

    name: str
    real: float
    synthetic_mean: float
    synthetic_std: float
    synthetic_p2_5: float
    synthetic_p97_5: float
    real_percentile: float
    n_paths: int

    @property
    def real_inside_interval(self) -> bool:
        """True if the real value falls inside the synthetic 95% interval."""
        return self.synthetic_p2_5 <= self.real <= self.synthetic_p97_5

    def as_dict(self) -> dict[str, object]:
        return {
            "statistic": self.name,
            "real": self.real,
            "synthetic_mean": self.synthetic_mean,
            "synthetic_std": self.synthetic_std,
            "synthetic_p2_5": self.synthetic_p2_5,
            "synthetic_p97_5": self.synthetic_p97_5,
            "real_percentile": self.real_percentile,
            "real_inside_95pc_interval": self.real_inside_interval,
            "n_paths": self.n_paths,
        }


@dataclass
class MomentReport:
    """The full comparison, plus the summary a reader should look at first."""

    comparisons: list[MomentComparison] = field(default_factory=list)

    @property
    def failures(self) -> list[MomentComparison]:
        """Statistics whose real value falls outside the synthetic 95% interval."""
        return [c for c in self.comparisons if not c.real_inside_interval]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_statistics": len(self.comparisons),
            "n_outside_95pc_interval": len(self.failures),
            "outside": [c.name for c in self.failures],
            "comparisons": [c.as_dict() for c in self.comparisons],
        }


# --- the statistics -------------------------------------------------------------


def absolute_autocorrelation(returns: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation of |returns| at ``lag`` — the standard volatility-clustering measure."""
    return _autocorrelation(np.abs(returns), lag)


def _autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag < 1 or lag >= series.shape[0]:
        raise MomentError(f"lag {lag} is out of range for {series.shape[0]} observations")
    centred = series - series.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centred[:-lag], centred[lag:]) / denominator)


def _skewness(series: np.ndarray) -> float:
    centred = series - series.mean()
    variance = float(np.mean(centred**2))
    if variance == 0.0:
        return 0.0
    return float(np.mean(centred**3) / variance**1.5)


def _excess_kurtosis(series: np.ndarray) -> float:
    centred = series - series.mean()
    variance = float(np.mean(centred**2))
    if variance == 0.0:
        return 0.0
    return float(np.mean(centred**4) / variance**2 - 3.0)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative path, as a positive fraction.

    Path-dependent, so no moment-matching argument guarantees a bootstrap reproduces it. That is
    precisely why it is here: it is the statistic a stress-testing method most needs to get right
    and is least entitled to assume.
    """
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


#: The statistic battery. Keeping it as data rather than a hard-coded sequence of calls means the
#: set is visible in one place and cannot quietly grow to include whichever statistic happened to
#: pass.
STATISTICS: tuple[tuple[str, object], ...] = (
    ("mean", lambda r: float(np.mean(r))),
    ("std", lambda r: float(np.std(r, ddof=1))),
    ("skewness", _skewness),
    ("excess_kurtosis", _excess_kurtosis),
    ("autocorr_lag1", lambda r: _autocorrelation(r, 1)),
    ("abs_autocorr_lag1", lambda r: absolute_autocorrelation(r, 1)),
    ("abs_autocorr_lag5", lambda r: absolute_autocorrelation(r, 5)),
    ("abs_autocorr_lag21", lambda r: absolute_autocorrelation(r, 21)),
    ("max_drawdown", max_drawdown),
)


def compare_moments(
    real_returns: np.ndarray,
    index_paths: np.ndarray,
) -> MomentReport:
    """Compare each statistic on the real series against its synthetic distribution.

    ``index_paths`` is the ``(n_paths, n_obs)`` integer array from
    :mod:`src.stress.crr`; each row is applied to ``real_returns`` to form one synthetic path.
    """
    real_returns = np.asarray(real_returns, dtype=np.float64).ravel()
    index_paths = np.asarray(index_paths)
    if index_paths.ndim != 2:
        raise MomentError(f"index_paths must be 2-D, got shape {index_paths.shape}")
    if index_paths.shape[1] != real_returns.shape[0]:
        raise MomentError(
            f"paths are {index_paths.shape[1]} long but the real series is "
            f"{real_returns.shape[0]}; they must match"
        )

    synthetic = real_returns[index_paths]     # (n_paths, n_obs)
    report = MomentReport()
    for name, function in STATISTICS:
        real_value = float(function(real_returns))  # type: ignore[operator]
        drawn = np.array([float(function(path)) for path in synthetic])  # type: ignore[operator]
        report.comparisons.append(
            MomentComparison(
                name=name,
                real=real_value,
                synthetic_mean=float(np.mean(drawn)),
                synthetic_std=float(np.std(drawn, ddof=1)),
                synthetic_p2_5=float(np.percentile(drawn, 2.5)),
                synthetic_p97_5=float(np.percentile(drawn, 97.5)),
                real_percentile=float(np.mean(drawn <= real_value) * 100.0),
                n_paths=int(index_paths.shape[0]),
            )
        )
    return report
