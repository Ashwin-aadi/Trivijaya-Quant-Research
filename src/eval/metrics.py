"""Performance statistics for a return series.

Deliberately plain. Every figure here is reported alongside its sample size elsewhere in the
project, because an annualised Sharpe computed from forty observations is not the same claim as
one computed from a thousand, and the number alone does not say which it is.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Sessions per year used to annualise daily statistics.
#
# 252 rather than the ~250 NSE actually trades, because the Deflated Sharpe machinery in
# src/audit/stat.py is defined against the Bailey & Lopez de Prado convention, which assumes 252.
# Mixing a 250-annualised Sharpe into a 252-defined DSR reintroduces exactly the boundary mismatch
# this module was standardised to remove. One convention, everywhere.
TRADING_DAYS_PER_YEAR = 252

# Annualised volatility below this is floating-point residue, not risk. Set far under any real
# strategy's volatility (the quietest cash-like book still moves by basis points) and far above
# the ~1e-16 noise floor a constant series leaves behind.
NEGLIGIBLE_VOLATILITY = 1e-12


def annualised_return(returns: Sequence[float]) -> float:
    """Arithmetic mean return, annualised. Returns 0.0 for an empty series.

    **Arithmetic, not geometric, and the choice is deliberate.** A geometric annualisation answers
    "what did capital actually compound to", which is the right question for a performance report.
    But this figure feeds :func:`sharpe_ratio`, and the Deflated Sharpe in ``src/audit/stat.py``
    is defined on an arithmetic per-observation Sharpe. Two conventions differing by double digits
    on the same series is precisely the seam through which a wrong number reaches the deflation
    machinery, so the repository carries one convention only. Use :func:`total_return` when the
    question really is how much money was made.
    """
    if not returns:
        return 0.0
    return sum(returns) / len(returns) * TRADING_DAYS_PER_YEAR


def total_return(returns: Sequence[float]) -> float:
    """Cumulative growth over the whole series, as a fraction. Not annualised.

    Kept separate from :func:`annualised_return` so that "how much did it compound" and "what goes
    into the Sharpe" are different names rather than one name meaning two things.
    """
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    return growth - 1.0


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
    # Tolerance, not exact equality. A mathematically constant series does not generally produce
    # a variance of exactly zero in floating point — [0.01] * 250 leaves a residue near 1e-16 —
    # and dividing by that residue yields a Sharpe around 1e16. Anything below this threshold is
    # numerical noise rather than a real risk measurement, so the ratio is undefined and reported
    # as zero.
    if volatility < NEGLIGIBLE_VOLATILITY:
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
