# Ground truth — Group C leaky fixtures

Six strategies, two per category, each with exactly one deliberate data-leakage defect. Written
independently of every other fixture in this benchmark.

## point_in_time_bypass_a.py

**Category:** point_in_time_bypass

`OpeningRangeBreakout.__init__` stores the full price panel it is handed (`full_panel`) as
`self._panel`, unfiltered and untruncated. `generate()` calls the helper `_opening_prices`, which
queries `self._panel` for rows where `session_date == as_of` to get each symbol's opening print
*on the decision date itself*, and uses that price both to decide which names count as breakouts
and to size the resulting positions. Per the engine's contract, `view.as_of` is the date the
order fills at the open — so this is the fill price itself, read before the fill happens. A
reviewer skimming `generate()` would see `view.history(...)` and `view.symbols` used correctly
everywhere else, and could easily read the `self._panel` lookup as an incidental convenience
method rather than a second, unguarded data channel.

## point_in_time_bypass_b.py

**Category:** point_in_time_bypass

`_smoothed_volume_panel()` (module level) reads the entire volume history straight off disk via
`pl.read_parquet`, caches it in the module global `_SMOOTHED_VOLUME`, and smooths it with
`pl.col("volume").rolling_mean(window_size=5, center=True)`. `generate()` in
`LiquidityFilteredReversal` consults this cache at `view.as_of` to apply its ADV floor. Two
things compound: the cache is never truncated to any `as_of` (it is loaded once, globally, for
the whole file's history), and `center=True` means the value attributed to a given session
average in two sessions *after* it as well as two before. The liquidity filter on any given day
therefore depends on volume two sessions in the future. It is easy to miss because the return
computation (`weekly`) is built correctly from `view.history(5)`, and the leak hides behind a
single keyword argument (`center=True`) deep inside an otherwise ordinary-looking smoothing call.

## future_dependent_ordering_a.py

**Category:** future_dependent_ordering

`QualityCompounderMomentum._select_compounders` (a staticmethod invoked once from `__init__` on
the full-period `universe_history` argument) computes each symbol's total return as
`adj_close.last() / adj_close.first() - 1.0` over the *entire* panel, sorts descending, and keeps
the top `basket_size` as `self._core_basket` — a fixed set used for the strategy's whole life.
The eligible trading universe is therefore chosen by which stocks turned out, by the end of the
study period, to have compounded the most; that is future information relative to any early
decision date. `generate()` itself is point-in-time clean (`view.history(63)`, `view.symbols`),
so a reviewer checking only the per-session rebalancing logic would find nothing wrong — the
defect is entirely inside a one-time constructor call dressed up as building a "pre-vetted"
investable list.

## future_dependent_ordering_b.py

**Category:** future_dependent_ordering

`_best_horizons()` (module level, memoized in `_BEST_HORIZON`) reads the full price panel from
disk and, for each candidate window in `_CANDIDATE_WINDOWS`, takes
`panel.group_by("symbol", maintain_order=True).tail(window)` — the last `window` rows of the
*entire dataset*, not of anything bounded by a decision date — computes each window's total
return, and keeps whichever window is the argmax per symbol as that symbol's momentum horizon.
`AdaptiveHorizonRotation.generate()` then uses `view.history(window)` with that chosen horizon,
which looks properly point-in-time on its own. The defect is that the horizon choice itself was
made by checking, in hindsight, which lookback would have looked best on the tail of the whole
history (which reaches to the dataset's final date, not to `view.as_of`) — an argmax over the
entire series, exactly the ordering-depends-on-the-outcome pattern. It is easy to miss because
the suspicious computation lives in a module-level cache function far from `generate()`.

## snooped_parameter_a.py

**Category:** snooped_parameter

`ZScoreReversion._calibrate_threshold` (staticmethod, run once from `__init__` against the
constructor's `calibration_data`, a full-period panel) builds a `fwd_close` column via
`adj_close.shift(-5).over("symbol")` — five sessions *forward* — and then loops over
`_CANDIDATE_THRESHOLDS`, scoring each by the mean forward return of names that crossed it, and
keeps the threshold with the best score as `self._threshold`. The visible for-loop makes the
search obvious, but the score it optimizes is computed from each candidate's future outcome over
the full sample, so the single constant threshold used for every subsequent trading decision was
picked by seeing what already happened next. It is easy to miss because `generate()`'s live
decision path only touches `view.history(...)` and `view.latest_close()` — nothing in the
day-to-day trading logic looks wrong — and the forward shift is buried as one intermediate column
inside an otherwise reasonable-looking calibration routine.

## snooped_parameter_b.py

**Category:** snooped_parameter

`_tuned_window()` (module level, called once from `MovingAverageCrossover.__init__`) reads the
full reference panel from disk and evaluates `_score_window(panel, window)` — an in-sample
Sharpe ratio of the crossover rule — for every window in `_CANDIDATE_WINDOWS` via the list
comprehension `[(window, _score_window(panel, window)) for window in _CANDIDATE_WINDOWS]`, then
takes `max(scores, key=lambda pair: pair[1])[0]`. The moving-average window used for every single
day of live trading is therefore the one hyperparameter value that scored best against the full
panel, including dates far beyond any given decision date. A reviewer is likely to accept this
because the comprehension and `max()` read as ordinary, principled parameter selection ("don't
pick a round number by feel"), and because `generate()` itself only calls
`view.history(self._window)`, so the per-session trading logic is point-in-time clean — the only
issue is that the one-time tuning step was run against the whole panel rather than a strictly
historical slice available before backtesting began.
