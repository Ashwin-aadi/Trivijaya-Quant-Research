"""Market-impact primitives, and the diagnostics that say whether daily bars can identify them.

Two kinds of thing live here, and the distinction is the whole point of Phase 3.0.

**Measures** are computed from observables and assume nothing. Amihud illiquidity, traded value,
and the participation rate are measures: they are what the data says, and a reader may disagree
with their usefulness but not with their value.

**Models** map an order size to a price move. The square-root law is a model. It cannot be
observed from daily bars, because a daily bar reports the aggregate of every participant's trading
and never our own order in isolation. :func:`square_root_impact` therefore implements the standard
functional form and takes its coefficient as an argument — it does not, and this module does not
anywhere, claim to have calibrated that coefficient to Indian data.

The functions in the second half of this module exist to *measure the size of that gap* rather
than to assert it. They ask whether a daily-bar regression of price move on volume even recovers a
stable exponent, whether the move it explains is transient (which impact is) or permanent (which
information is), and whether the region of order sizes a capacity study actually cares about lies
inside the observed data or far outside it. Phase 3.0 halts on their answers.

References
----------
Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects."
    *Journal of Financial Markets* 5(1), 31-56. ILLIQ is equation (1).
Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). "Direct estimation of equity market impact."
    *Risk* 18(7), 58-62. The square-root form, calibrated there on US institutional order data --
    that is, on exactly the data this repository does not have.
Campbell, J., Grossman, S., Wang, J. (1993). "Trading volume and serial correlation in stock
    returns." *Quarterly Journal of Economics* 108(4), 905-939. The reversal regression used by
    :func:`reversal_betas` to separate a transient move from a permanent one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.common.exceptions import DataIntegrityError

#: Trailing sessions over which "normal" volume is measured. Matches
#: ``constraints.adv_window_sessions`` in config.yaml, and is passed in rather than read here so
#: this module stays pure.
DEFAULT_ADV_WINDOW = 21


# --------------------------------------------------------------------------------------------
# Measures. These assume nothing beyond the arithmetic.
# --------------------------------------------------------------------------------------------


def participation_rate(order_value_inr: float, traded_value_inr: float) -> float:
    """Fraction of a session's traded value that an order of this size would represent.

    Raises on a non-positive denominator rather than returning infinity: a session with no traded
    value is one where the order could not have been executed at all, and a silent ``inf`` would
    propagate into a capacity curve as a merely large number.
    """
    if traded_value_inr <= 0:
        raise DataIntegrityError(
            f"participation is undefined against traded value {traded_value_inr}; the session had "
            "no trading, so the order was not executable rather than expensively executable"
        )
    return order_value_inr / traded_value_inr


def square_root_impact(
    participation: float,
    volatility: float,
    coefficient: float,
) -> float:
    """Almgren et al. (2005) square-root impact: ``coefficient * volatility * sqrt(participation)``.

    **The coefficient is an assumption supplied by the caller, not a calibration.** Almgren et al.
    fit it to a proprietary record of institutional orders, in which the order and its own price
    path are separately observed. Daily bars contain no such separation, so nothing in this
    repository estimates this number from Indian data. It must never be described as if it did.

    The functional form itself is standard and is what makes the model useful even uncalibrated:
    impact is concave in size, so a capacity curve built on it bends the right way regardless of
    where the level sits.
    """
    if participation < 0:
        raise DataIntegrityError(f"participation must be non-negative, got {participation}")
    if volatility < 0:
        raise DataIntegrityError(f"volatility must be non-negative, got {volatility}")
    return coefficient * volatility * math.sqrt(participation)


def amihud_illiquidity(returns: pl.Series, traded_value_inr: pl.Series) -> float:
    """Amihud (2002) eq. (1): the mean of ``|return| / traded value`` over available sessions.

    Interpreted as the price response per rupee traded. Sessions with no traded value are dropped
    rather than treated as infinitely illiquid, and the count that survived is the caller's
    responsibility to report -- an ILLIQ computed over eleven sessions and one computed over a
    thousand are not comparable, and nothing in the returned scalar records which it was.
    """
    frame = pl.DataFrame({"r": returns, "v": traded_value_inr}).drop_nulls()
    frame = frame.filter(pl.col("v") > 0)
    if frame.height == 0:
        raise DataIntegrityError("no session with positive traded value; ILLIQ is undefined")
    ratio = (frame["r"].abs() / frame["v"]).to_numpy()
    return float(np.mean(ratio))


def add_daily_measures(
    panel: pl.DataFrame, *, adv_window: int = DEFAULT_ADV_WINDOW
) -> pl.DataFrame:
    """Attach return, trailing ADV, participation and the Amihud term to a symbol-day panel.

    The trailing average traded value is computed over sessions **strictly before** the row's own,
    for the same reason the backtester enforces it: a normalisation that includes today's volume
    knows today's volume, and every ratio built on it is then partly a function of itself.
    """
    required = {"session_date", "symbol", "adj_close", "turnover_inr", "volume"}
    missing = required - set(panel.columns)
    if missing:
        raise DataIntegrityError(f"panel is missing columns {sorted(missing)}")

    return (
        panel.sort(["symbol", "session_date"])
        .with_columns(
            ret=(pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0),
            adv_inr=pl.col("turnover_inr")
            .shift(1)
            .rolling_mean(window_size=adv_window, min_samples=adv_window)
            .over("symbol"),
        )
        .with_columns(
            abs_ret=pl.col("ret").abs(),
            # Volume relative to its own recent normal. This is the regressor a daily-bar impact
            # study is forced to use, and the diagnostics below measure what it can support.
            rel_value=pl.when(pl.col("adv_inr") > 0)
            .then(pl.col("turnover_inr") / pl.col("adv_inr"))
            .otherwise(None),
        )
        .with_columns(
            amihud_term=pl.when(pl.col("turnover_inr") > 0)
            .then(pl.col("ret").abs() / pl.col("turnover_inr"))
            .otherwise(None),
        )
    )


# --------------------------------------------------------------------------------------------
# Diagnostics. These measure whether a model can be identified, not what the model says.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ElasticityFit:
    """One symbol's fit of ``|return| = a * (relative traded value)^delta``, in logs.

    ``delta`` is the quantity the square-root law predicts to be 0.5. It is reported per symbol
    rather than pooled because a single pooled number hides whether the exponent is a property of
    the market or an average over symbols that disagree.
    """

    symbol: str
    delta: float
    intercept: float
    r_squared: float
    n_sessions: int


def fit_elasticity(
    frame: pl.DataFrame,
    *,
    min_sessions: int,
) -> list[ElasticityFit]:
    """Per-symbol OLS of ``log |return|`` on ``log(relative traded value)``.

    Zero returns are dropped, not floored: a floor is a free parameter that moves the estimated
    exponent, and a session that closed exactly flat carries no information about the size of a
    move. The number of sessions surviving that drop is recorded on every fit so the reader can
    see which symbols were estimated on thin evidence.
    """
    fits: list[ElasticityFit] = []
    usable = frame.filter(
        pl.col("abs_ret").is_not_null()
        & (pl.col("abs_ret") > 0)
        & pl.col("rel_value").is_not_null()
        & (pl.col("rel_value") > 0)
    )
    for (symbol,), group in usable.group_by(["symbol"], maintain_order=True):
        if group.height < min_sessions:
            continue
        x = np.log(group["rel_value"].to_numpy())
        y = np.log(group["abs_ret"].to_numpy())
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total = y - y.mean()
        r_squared = 1.0 - float(residual @ residual) / float(total @ total)
        fits.append(
            ElasticityFit(
                symbol=str(symbol),
                delta=float(coefficients[1]),
                intercept=float(coefficients[0]),
                r_squared=r_squared,
                n_sessions=group.height,
            )
        )
    return fits


@dataclass(frozen=True)
class ReversalFit:
    """One symbol's regression of the next ``horizon`` sessions' return on today's.

    A price move caused by an order is *transient*: it is the cost of demanding liquidity, and it
    decays once the demand stops. A move caused by information is *permanent*. A negative ``beta``
    is therefore evidence of impact and a ``beta`` near zero is evidence that what the daily
    volume-return relation measures is news. This is the test that decides whether a daily-bar
    impact estimate means what its name says.

    ``standard_error`` is carried because a null is not a result without it. "We found no reversal"
    and "we could not have found one" are different claims, and only :func:`minimum_detectable_beta`
    can tell them apart.
    """

    symbol: str
    beta: float
    standard_error: float
    r_squared: float
    n_sessions: int


def reversal_betas(
    frame: pl.DataFrame,
    *,
    horizon: int,
    min_sessions: int,
    high_participation_quantile: float | None = None,
) -> list[ReversalFit]:
    """Campbell-Grossman-Wang reversal regression, optionally restricted to heavy-volume sessions.

    Uses returns after the session being described, which is legitimate here and nowhere else in
    this repository: the object being estimated is a property of the market, computed once and
    never fed to a strategy. Any use of these coefficients inside a trading rule would be leakage.
    """
    fits: list[ReversalFit] = []
    forward = frame.sort(["symbol", "session_date"]).with_columns(
        fwd=(
            pl.col("adj_close").shift(-horizon).over("symbol") / pl.col("adj_close") - 1.0
        )
    )
    for (symbol,), group in forward.group_by(["symbol"], maintain_order=True):
        usable = group.drop_nulls(["ret", "fwd", "rel_value"])
        if high_participation_quantile is not None:
            cut = usable["rel_value"].quantile(high_participation_quantile)
            if cut is None:
                continue
            usable = usable.filter(pl.col("rel_value") >= cut)
        if usable.height < min_sessions:
            continue
        x = usable["ret"].to_numpy()
        y = usable["fwd"].to_numpy()
        design = np.column_stack([np.ones_like(x), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total = y - y.mean()
        denominator = float(total @ total)
        explained = (
            1.0 - float(residual @ residual) / denominator if denominator > 0 else 0.0
        )
        fits.append(
            ReversalFit(
                symbol=str(symbol),
                beta=float(coefficients[1]),
                standard_error=_slope_standard_error(x, residual),
                r_squared=explained,
                n_sessions=usable.height,
            )
        )
    return fits


def _slope_standard_error(x: np.ndarray, residual: np.ndarray) -> float:
    """Textbook OLS slope standard error, ``s / sqrt(sum (x - xbar)^2)``, with 2 parameters fit."""
    degrees_of_freedom = x.size - 2
    spread = float(((x - x.mean()) ** 2).sum())
    if degrees_of_freedom <= 0 or spread <= 0:
        return float("nan")
    residual_variance = float(residual @ residual) / degrees_of_freedom
    return math.sqrt(residual_variance / spread)


@dataclass(frozen=True)
class Detectability:
    """How large a reversal the data could have found, against how large a one it did find.

    **A null result without this is not a result.** "No reversal was detected" is compatible with
    two very different worlds: one where the test was sharp and there is genuinely nothing there,
    and one where the test could not have seen a reversal of any plausible size. Reporting the
    minimum detectable effect distinguishes them, and it is the difference between a publishable
    negative finding and an absence of evidence dressed up as evidence of absence.

    ``pooled_beta`` combines every symbol by inverse-variance weighting, which is the right summary
    when the per-symbol estimates share a target and differ only in precision.
    """

    #: Smallest |beta| the median symbol could reject the null against, at the stated power.
    median_minimum_detectable_beta: float
    #: The same for the pooled estimate, which is far sharper because it uses every symbol.
    pooled_minimum_detectable_beta: float
    pooled_beta: float
    pooled_standard_error: float
    #: Every symbol counted once, regardless of how precisely it was estimated. Reported alongside
    #: the pooled figure because the two answer different questions, and when they disagree that
    #: disagreement is itself the finding: inverse-variance weighting concentrates on whichever
    #: symbols happen to be most precisely estimated, so a pooled estimate can describe a handful of
    #: names while the unweighted one describes the population. Neither is "the" answer.
    unweighted_mean_beta: float
    unweighted_standard_error: float
    power: float
    alpha: float
    n_symbols: int

    @property
    def estimates_disagree(self) -> bool:
        """True when pooling and equal weighting do not even agree on the sign of the effect."""
        return self.pooled_beta * self.unweighted_mean_beta < 0


def minimum_detectable_beta(
    fits: list[ReversalFit],
    *,
    power: float = 0.80,
    alpha: float = 0.05,
) -> Detectability:
    """Bound the reversal that would have been found had one been there.

    The minimum detectable effect for a two-sided test is ``(z_{1-alpha/2} + z_{power}) * se``.
    Both critical values are hard-coded for the conventional 5%/80% pair rather than computed, so
    this function has no dependency on a statistics package and the arithmetic is inspectable; any
    other pair raises rather than silently using the wrong multiplier.
    """
    if (power, alpha) != (0.80, 0.05):
        raise DataIntegrityError(
            f"only the conventional power=0.80, alpha=0.05 pair is implemented; got "
            f"power={power}, alpha={alpha}. Add the critical values explicitly rather than "
            "approximating them."
        )
    multiplier = 1.959964 + 0.841621  # z_{0.975} + z_{0.80}

    errors = np.array([f.standard_error for f in fits], dtype=float)
    betas = np.array([f.beta for f in fits], dtype=float)
    usable = np.isfinite(errors) & (errors > 0) & np.isfinite(betas)
    if not usable.any():
        raise DataIntegrityError("no fit carries a usable standard error; detectability undefined")
    errors, betas = errors[usable], betas[usable]

    # Inverse-variance (fixed-effect) pooling: weight each symbol by its precision.
    weights = 1.0 / errors**2
    pooled = float((weights * betas).sum() / weights.sum())
    pooled_error = float(math.sqrt(1.0 / weights.sum()))

    # Equal weighting, treating each symbol as one observation of a population of betas. Its
    # standard error is the spread across symbols, not the precision within any one of them.
    unweighted = float(betas.mean())
    unweighted_error = (
        float(betas.std(ddof=1) / math.sqrt(betas.size)) if betas.size > 1 else float("nan")
    )

    return Detectability(
        median_minimum_detectable_beta=float(multiplier * np.median(errors)),
        pooled_minimum_detectable_beta=multiplier * pooled_error,
        pooled_beta=pooled,
        pooled_standard_error=pooled_error,
        unweighted_mean_beta=unweighted,
        unweighted_standard_error=unweighted_error,
        power=power,
        alpha=alpha,
        n_symbols=int(usable.sum()),
    )


@dataclass(frozen=True)
class ExtrapolationGap:
    """How far the order sizes a capacity study asks about sit outside the observed data.

    A daily-bar impact regression is fitted where relative traded value is of order 1, because that
    is what a day's total volume is. A capacity study asks what happens at a participation of a
    fraction of a percent. The ratio between the two is the extrapolation the model is being asked
    to perform, and it is reported in orders of magnitude because that is the honest unit.
    """

    target_participation: float
    observed_p01: float
    observed_median: float
    orders_of_magnitude: float
    fraction_below_target: float
    n_symbol_days: int


def extrapolation_gap(frame: pl.DataFrame, *, target_participation: float) -> ExtrapolationGap:
    """Compare the region a capacity curve is evaluated in against the region the data occupies."""
    values = frame["rel_value"].drop_nulls()
    values = values.filter(values > 0)
    if values.len() == 0:
        raise DataIntegrityError("no positive relative traded values; the gap is undefined")
    p01 = values.quantile(0.01)
    median = values.quantile(0.5)
    if p01 is None or median is None:
        raise DataIntegrityError("relative traded value has no quantiles; the gap is undefined")
    below = float((values < target_participation).sum()) / float(values.len())
    return ExtrapolationGap(
        target_participation=target_participation,
        observed_p01=float(p01),
        observed_median=float(median),
        orders_of_magnitude=math.log10(float(median) / target_participation),
        fraction_below_target=below,
        n_symbol_days=values.len(),
    )
