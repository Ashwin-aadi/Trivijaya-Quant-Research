"""Performance statistics for a return series.

Deliberately plain. Every figure here is reported alongside its sample size elsewhere in the
project, because an annualised Sharpe computed from forty observations is not the same claim as
one computed from a thousand, and the number alone does not say which it is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# NSE trades roughly 250 sessions a year; used to annualise daily statistics.
TRADING_DAYS_PER_YEAR = 250


def annualised_return(returns: Sequence[float]) -> float:
    """Geometric mean return, annualised. Returns 0.0 for an empty series."""
    if not returns:
        return 0.0
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    if growth <= 0:
        return -1.0
    return float(growth ** (TRADING_DAYS_PER_YEAR / len(returns))) - 1.0


def annualised_volatility(returns: Sequence[float]) -> float:
    """Standard deviation of returns, annualised."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    The risk-free rate defaults to zero and is stated explicitly at every call site rather than
    quietly assumed, since an Indian backtest run against a ~6-7% policy rate is a materially
    different claim from one run against zero.
    """
    if len(returns) < 2:
        return 0.0
    volatility = annualised_volatility(returns)
    if volatility == 0:
        return 0.0
    return (annualised_return(returns) - risk_free_rate) / volatility


def max_drawdown(returns: Sequence[float]) -> float:
    """Largest peak-to-trough decline in cumulative equity, as a negative fraction."""
    peak = 1.0
    equity = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def tracking_error(returns: Sequence[float], benchmark: Sequence[float]) -> float:
    """Annualised standard deviation of the return difference against a benchmark."""
    if len(returns) != len(benchmark):
        raise ValueError(
            f"series lengths differ: {len(returns)} returns vs {len(benchmark)} benchmark"
        )
    differences = [r - b for r, b in zip(returns, benchmark, strict=True)]
    return annualised_volatility(differences)


def summarise(returns: Sequence[float]) -> dict[str, float]:
    """All headline statistics plus the sample size they rest on."""
    return {
        "n_sessions": float(len(returns)),
        "annualised_return": annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }
