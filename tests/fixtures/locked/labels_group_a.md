# Ground truth — leaky fixtures, group A

Six strategies, each containing exactly one deliberate leakage defect. Written independently,
without reference to any other fixture already in this benchmark.

## `future_indexing_a.py` — `BreakoutContinuation`
**Category:** future_indexing

The defect is in `_with_breakout_flags`: `pl.col("close").shift(-1).over("symbol").alias("next_close")`,
whose result (`next_close`) is then required to sit above `rolling_high` to set the `held` flag
that gates entries in `generate()`. For the most recent row visible to a decision (the row the
strategy actually trades on), the session one row later is the decision date itself, so the
"confirmation" the code demands is the close of a bar dated at the moment the trade is being
placed. A reviewer skimming this is likely to read `held` as ordinary risk management ("only buy
confirmed breakouts") rather than notice that "confirmation" is defined using the very next bar,
because the negative shift is nested two operations deep inside a small helper alongside a
legitimate backward-looking rolling max.

## `future_indexing_b.py` — `OversoldBounce`
**Category:** future_indexing

The defect is in `_distance_from_recent_low`: the frame is reversed with `.reverse()`, a
`rolling_min` is computed over that reversed row order, and the frame is reversed back before the
`distance_from_low` column is attached. Because a rolling window only accumulates over the
physical row order it is given, computing it after reversing to descending date order makes
`recent_low` for any date `d` equal to the minimum close over `d` and the sessions *after* it
within the window, not the sessions before it — a "recent low" that is actually a forward-looking
low. A reviewer might miss it because the accompanying comment offers a plausible-sounding
justification for the reversal, and reverse-then-restore is a shape that looks like careful
engineering rather than an inverted rolling window.

## `survivorship_selection_a.py` — `LargeCapQualityTilt`
**Category:** survivorship_selection

The defect is the module-level `_UNIVERSE` tuple — a hardcoded list of today's well-known large
caps — used in `generate()` via `frame.filter(pl.col("symbol").is_in(_UNIVERSE))` in place of the
point-in-time-correct `view.symbols` that `MarketView` already supplies. Any stock that was
removed from the index at some point during the study period is simply absent from `_UNIVERSE`
and can never be selected, even on early dates when it was a genuine, tradable constituent. A
reviewer might miss it because a fixed list of "well established large caps" reads as an ordinary
liquidity filter, and the code never touches `view.symbols` at all, so there is no visible
contrast between a correct and incorrect universe to draw attention to the choice.

## `survivorship_selection_b.py` — `LowVolatilityTilt`
**Category:** survivorship_selection

The defect is in `__init__`, which calls `_eligible_symbols(index_membership)` — filtering
`index_membership.filter(pl.col("exit_date").is_null())` — exactly once at construction and
stores the result as `self._eligible` for reuse across every `generate()` call for the life of the
backtest. The reference table does carry per-constituent `entry_date`/`exit_date` fields that
would support a correct per-`view.as_of` lookup, but the code never performs that lookup; it uses
"still a member as of the table's snapshot" instead. Any symbol removed from the index before the
present day is excluded from every session, including early dates when it was legitimately
tradable. It is easy to miss because the table genuinely is point-in-time-capable and the code
reads as though it is using it correctly — the omission is that the eligibility check runs once
at construction rather than once per `generate()` call against `view.as_of`.

## `full_sample_fit_a.py` — `QualityComposite`
**Category:** full_sample_fit

The defect is in `__init__`, which calls `_momentum_stats(panel, _MOMENTUM_WINDOW)` on the full
panel handed to the constructor and stores the resulting per-symbol `mean_momentum` /
`std_momentum` as `self._stats`, fit once over data spanning the entire study period. `generate()`
then joins these fixed statistics onto every session's ranking, so a decision made near the start
of the backtest is standardized against dispersion and central-tendency figures that are partly
computed from years not yet observed at that decision date. It is easy to miss because fitting a
normalization "once for efficiency" is completely standard-looking practice, and the stated
rationale (comparability across market eras) gives an entirely plausible reason to want a stable,
un-recomputed yardstick — which is exactly what makes the full-sample fit look intentional rather
than leaky.

## `full_sample_fit_b.py` — `VolManagedMomentum`
**Category:** full_sample_fit

The defect is in `__init__`, which computes `self._high_vol_cutoff` as the 90th percentile of
`_index_realized_vol(panel, _VOL_WINDOW)` over the complete panel passed to the constructor — the
whole study period — and stores it as a fixed threshold. `_current_scale` then compares each
session's live, correctly point-in-time volatility (derived from `view.history()`) against this
stale full-sample cutoff for the entire backtest. Because the panel spans the full period, the
threshold used to classify, say, a 2016 session as "unusually high volatility" is calibrated partly
using volatility readings from crises that had not yet happened, such as the 2020 crash. It is
easy to miss because the per-session volatility calculation itself is genuinely point-in-time
correct — only the threshold it is compared against is stale — so most of the method reads as
careful, and the rationale frames the fixed cutoff as sound risk management rather than a fit
boundary violation.
