"""Strategy characteristics: what a strategy *is*, described without running the stress suite.

Phase 2.2 asks whether fragility can be predicted from a strategy's structure rather than measured
by five and a half hours of counterfactual backtesting. That requires describing a strategy by
properties available the moment it has been run once on real history — turnover, how long it holds
a name, how concentrated the book is, how fast the book decays, and what factors it loads on.

Everything here is computed from a single real-panel backtest. Nothing uses a synthetic path, and
nothing uses the target, so a feature cannot leak the quantity being predicted.

Two properties need positions, not returns, and cannot be recovered from an equity curve at all:

``holding_period``
    Held name-sessions divided by entries. A strategy that buys a name and keeps it for a quarter
    and one that churns it daily can produce identical return series.

``concentration``
    Herfindahl index over absolute weights, plus the effective number of holdings ``1/HHI``. Two
    books with the same gross exposure can be one bet or fifty.

``book_autocorrelation`` is the closest available stand-in for the charter's "signal-decay profile":
the cosine similarity between the weight vector today and the weight vector *k* sessions later. A
fast-decaying signal rebuilds its book from scratch; a slow one barely moves it. It is a property of
the realised book rather than of the underlying alpha forecast, which the engine never stores, and
that limitation is stated rather than glossed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Horizons in sessions at which the book's persistence is measured: about a week, a month, a
#: quarter. Chosen to bracket the holding periods the corpus actually exhibits.
DECAY_HORIZONS = (5, 21, 63)

#: Below this absolute weight a name is treated as not held. Guards against a strategy that leaves
#: a 1e-18 residual in a name it has exited, which would otherwise register as a continuous holding
#: and inflate the holding period without bound.
HELD_THRESHOLD = 1e-8


def _herfindahl(weights: np.ndarray) -> float:
    """Sum of squared *shares* of gross exposure. 1.0 is one name; 1/n is n equal names."""
    gross = float(np.abs(weights).sum())
    if gross <= HELD_THRESHOLD:
        return float("nan")
    shares = np.abs(weights) / gross
    return float(np.square(shares).sum())


def concentration(books: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean Herfindahl, effective breadth, and largest single weight across sessions.

    Sessions holding nothing contribute no Herfindahl — a cash book has no concentration, and
    scoring it as either maximally concentrated or maximally diversified would be an invention.
    They still contribute to ``cash_session_rate``, which is where their information belongs.
    """
    hhi: list[float] = []
    largest: list[float] = []
    holdings: list[int] = []
    cash_sessions = 0
    for book in books:
        weights = np.array(
            [w for w in book.values() if abs(w) > HELD_THRESHOLD], dtype=float
        )
        if weights.size == 0:
            cash_sessions += 1
            continue
        hhi.append(_herfindahl(weights))
        largest.append(float(np.abs(weights).max() / np.abs(weights).sum()))
        holdings.append(int(weights.size))

    if not hhi:
        return {
            "mean_herfindahl": float("nan"),
            "effective_holdings": float("nan"),
            "mean_n_holdings": 0.0,
            "mean_largest_weight_share": float("nan"),
            "cash_session_rate": 1.0,
        }
    mean_hhi = float(np.mean(hhi))
    return {
        "mean_herfindahl": mean_hhi,
        "effective_holdings": 1.0 / mean_hhi,
        "mean_n_holdings": float(np.mean(holdings)),
        "mean_largest_weight_share": float(np.mean(largest)),
        "cash_session_rate": cash_sessions / len(books),
    }


def holding_period(books: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean sessions a name is held, from entries and held name-sessions.

    An entry is a session where a name is held and was not held the session before. Positions still
    open at the end of the sample are counted as held for as long as they were held, which biases
    the mean *downward* for slow strategies; that is the conservative direction for a feature meant
    to distinguish fast from slow, and it is stated here rather than corrected by extrapolation.
    """
    entries = 0
    held_sessions = 0
    previous: set[str] = set()
    for book in books:
        current = {s for s, w in book.items() if abs(w) > HELD_THRESHOLD}
        entries += len(current - previous)
        held_sessions += len(current)
        previous = current

    if entries == 0:
        # Either the strategy never held anything, or it held the same book from session one and
        # never traded. The two are distinguished by held_sessions, not by a fabricated mean.
        return {
            "mean_holding_period": float("nan") if held_sessions == 0 else float(len(books)),
            "n_entries": 0.0,
            "entries_per_session": 0.0,
        }
    return {
        "mean_holding_period": held_sessions / entries,
        "n_entries": float(entries),
        "entries_per_session": entries / len(books),
    }


def book_autocorrelation(
    books: Sequence[dict[str, float]], horizons: Sequence[int] = DECAY_HORIZONS
) -> dict[str, float]:
    """Cosine similarity between the book now and the book ``k`` sessions later, averaged.

    Cosine rather than Pearson: the quantity of interest is whether the *same names in the same
    proportions* are still held, and Pearson would first subtract a cross-sectional mean that has
    no meaning across a changing universe. Sessions where either book is empty are skipped, and the
    number of comparisons that survived is reported alongside each value.
    """
    out: dict[str, float] = {}
    for k in horizons:
        similarities: list[float] = []
        for i in range(len(books) - k):
            first, second = books[i], books[i + k]
            names = sorted(set(first) | set(second))
            if not names:
                continue
            a = np.array([first.get(n, 0.0) for n in names], dtype=float)
            b = np.array([second.get(n, 0.0) for n in names], dtype=float)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= HELD_THRESHOLD:
                continue
            similarities.append(float(a @ b / denom))
        out[f"book_similarity_{k}d"] = (
            float(np.mean(similarities)) if similarities else float("nan")
        )
        out[f"book_similarity_{k}d_n"] = float(len(similarities))
    return out


def turnover_profile(turnover: Sequence[float]) -> dict[str, float]:
    """Mean and dispersion of session turnover, and how often the strategy trades at all."""
    values = np.array(list(turnover), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean_turnover": float("nan"), "turnover_volatility": float("nan"),
                "trading_session_rate": float("nan")}
    return {
        "mean_turnover": float(values.mean()),
        "turnover_volatility": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "trading_session_rate": float((values > HELD_THRESHOLD).mean()),
    }


def joint_betas(
    returns: np.ndarray, factors: dict[str, np.ndarray]
) -> dict[str, float]:
    """One multiple regression of the strategy on all factors at once, with an intercept.

    This is the primary specification, per the PI ruling of 2026-08-02. A joint fit gives the
    *marginal* exposure to each factor holding the others fixed, which is what "factor exposure"
    means in the literature; univariate betas answer a different and weaker question, since a
    strategy with no independent momentum exposure will still show a large univariate momentum beta
    if momentum correlates with something it does hold.

    Solved by least squares on the design matrix rather than by normal equations: ``lstsq`` uses an
    SVD and returns a minimum-norm solution when the design is rank-deficient, instead of failing or
    — worse — returning enormous offsetting coefficients from an ill-conditioned inverse. The
    condition number is reported by :func:`design_diagnostics` so the reader can see how much to
    trust the individual coefficients, and ``r_squared`` records how much of the strategy the factor
    set explains in total, which is stable even where the split between factors is not.
    """
    names = sorted(factors)
    columns = [factors[name] for name in names]
    ok = np.isfinite(returns)
    for column in columns:
        ok &= np.isfinite(column)
    if int(ok.sum()) <= len(names) + 1:
        return {f"beta_{name}": float("nan") for name in names} | {"factor_r_squared": float("nan")}

    design = np.column_stack([np.ones(int(ok.sum()))] + [column[ok] for column in columns])
    target = returns[ok]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    total = float(np.square(target - target.mean()).sum())
    out = {f"beta_{name}": float(coefficients[position + 1]) for position, name in enumerate(names)}
    out["factor_r_squared"] = (
        1.0 - float(np.square(residual).sum()) / total if total > 0 else float("nan")
    )
    return out


def design_diagnostics(factors: dict[str, np.ndarray]) -> dict[str, float]:
    """How ill-conditioned the joint regression's design is, reported rather than assumed.

    The condition number is the ratio of largest to smallest singular value of the standardised
    factor matrix. Above roughly 30 the individual coefficients are unstable even though the fitted
    values are not — which is exactly the caveat a joint specification needs to carry.
    """
    names = sorted(factors)
    matrix = np.column_stack([factors[name] for name in names])
    finite = matrix[np.isfinite(matrix).all(axis=1)]
    standardised = (finite - finite.mean(axis=0)) / finite.std(axis=0, ddof=1)
    singular = np.linalg.svd(standardised, compute_uv=False)
    correlation = np.corrcoef(standardised, rowvar=False)
    off_diagonal = correlation[~np.eye(len(names), dtype=bool)]
    return {
        "n_factors": float(len(names)),
        "condition_number": float(singular.max() / singular.min()),
        "max_abs_pairwise_correlation": float(np.abs(off_diagonal).max()),
        "mean_abs_pairwise_correlation": float(np.abs(off_diagonal).mean()),
    }


def univariate_betas(
    returns: np.ndarray, factors: dict[str, np.ndarray]
) -> dict[str, float]:
    """One simple regression per factor: ``beta_f = Cov(r, f) / Var(f)``.

    Retained as a reported sensitivity, not as the primary specification. It is stable under
    collinearity — each coefficient depends on one factor only — at the cost of answering a weaker
    question: a strategy with no independent exposure to a factor still shows a large univariate
    beta to it whenever that factor correlates with something the strategy does hold.
    """
    out: dict[str, float] = {}
    for name, series in sorted(factors.items()):
        ok = np.isfinite(returns) & np.isfinite(series)
        if ok.sum() < 2:
            out[f"beta_{name}"] = float("nan")
            continue
        variance = float(np.var(series[ok], ddof=1))
        out[f"beta_{name}"] = (
            float(np.cov(returns[ok], series[ok], ddof=1)[0, 1] / variance)
            if variance > 0 else float("nan")
        )
    return out
