# Ground truth — locked leaky fixtures, group B

Six strategies, each written to contain exactly one deliberate data-leakage defect. No file is
marked internally; the defect location and mechanism for each is recorded here only.

---

## `full_sample_statistic_a.py` — `ValuationBandRotation`

**Category:** `full_sample_statistic`

**Where:** `__init__`, in the block that builds `stats = panel.group_by("symbol").agg(...)`. The
per-symbol `typical_level` (mean of `adj_close`) and `typical_spread` (std of `adj_close`) are
computed from the entire `panel` passed to the constructor, not from data restricted to before
each decision date.

**What it sees that it shouldn't:** on the very first tradable date, the strategy already knows
each stock's mean and standard deviation of price over the *whole* study period, including years
that haven't happened yet relative to that date. The z-score in `_z_scores` therefore compares
today's price against a "typical level" that was partly defined by the future.

**Why it's easy to miss:** the calibration step reads as ordinary, sensible code — "compute each
stock's typical trading range once so we don't recompute it every session" is exactly what a
careless-but-competent quant would write, and nothing about `group_by("symbol").agg(mean, std)`
looks unusual in isolation. The leak is entirely in what `panel` contains, which isn't visible
from the line itself.

---

## `full_sample_statistic_b.py` — `ExtremeMoveFade`

**Category:** `full_sample_statistic`

**Where:** `__init__`, in the block that builds `bands = returns.group_by("symbol").agg(...)`.
The per-symbol `lower_band` and `upper_band` are the `tail` and `1 - tail` quantiles of each
name's daily return, computed over the entire constructor-supplied `panel`.

**What it sees that it shouldn't:** the strategy is told, from day one, what the tails of each
stock's return distribution will turn out to be across the full multi-year sample, and uses that
future-informed threshold to decide whether *today's* move counts as "extreme" and worth fading.

**Why it's easy to miss:** it uses `.quantile()` rather than the more obviously-suspicious mean or
z-score, and the surrounding code (`_last_session_return`) correctly restricts itself to
`view.history(2)` — so a reviewer scanning `generate()` sees only point-in-time-safe operations.
The defect is upstream, in a helper that never touches `view` at all.

---

## `boundary_crossing_window_a.py` — `SmoothedTrendFollow`

**Category:** `boundary_crossing_window`

**Where:** `__init__`, the `rolling_mean(window_size=window, center=True)` call applied to
`adj_close` over the full constructor-supplied `panel`.

**What it sees that it shouldn't:** `center=True` means the smoothed value attached to any given
`(symbol, session_date)` is an average of sessions both before *and* after that date. For dates
early in the backtest, several of the sessions feeding that average lie in the future relative to
the decision being made on that date, so the "trend line" a stock is compared against on a given
day already encodes where its price was heading over the following sessions.

**Why it's easy to miss:** `center=True` is a completely ordinary smoothing choice — it's the
default mental model most people have for "a moving average" (symmetric, not trailing) — and the
bug never appears near `generate()`, which only reads from `view.latest_close()` and a
precomputed dict. Nothing in the live decision code looks unusual.

---

## `boundary_crossing_window_b.py` — `MonthlyStrengthTilt`

**Category:** `boundary_crossing_window`

**Where:** `__init__`, the `monthly = with_month.group_by(["symbol", "year", "month"]).agg(...)`
step, combined with the lookup in `generate()` keyed only by `(symbol, last_date.year,
last_date.month)`.

**What it sees that it shouldn't:** `monthly_strength` is the average daily return over an entire
calendar month, but it is looked up and used on *every* session that falls inside that month —
including the 1st, 3rd, or 10th trading day. On those early-month dates the value in use was
built from days later in that same month that have not happened yet, so the "this month's
strength" figure used mid-month already reflects the month's ending.

**Why it's easy to miss:** the code deliberately treats monthly strength as a slow-moving,
month-long "regime" variable — a legitimate modelling idea — so reusing the same value across
every day of the month reads as intentional design rather than a bug. The tell is subtle: the
aggregation window (a full month) does not match the granularity at which it's applied (daily).

---

## `target_in_features_a.py` — `FittedFactorBlend`

**Category:** `target_in_features`

**Where:** `__init__`, the line `feature_cols = [c for c in table.columns if c not in
("symbol", "session_date")]`. `table` (built by `_training_table`) contains the label column
`next_return` alongside the genuine features `momentum`, `volatility`, and `volume_trend`; the
exclusion list only drops the two identifier columns, so `next_return` is carried into
`feature_cols` and ends up in the design matrix `design` used to fit the regression, sitting
right next to the value it is trying to predict (`target = table["next_return"].to_numpy()`).

**What it sees that it shouldn't:** the fitted model is allowed to regress the label on itself
(among other things), so the coefficient on `next_return` can absorb essentially all of the fit,
degrading or masking whatever real signal the other three coefficients might otherwise have
carried.

**Why it's easy to miss:** the bug is a one-word omission in an exclusion list, the kind of thing
that is very easy to skim past. `_current_features`, used for live scoring in `generate()`, is a
separate and correctly-written function that never touches `next_return` at all, so testing the
live code path in isolation would not reveal anything wrong — the damage is already done at
fit time, inside `__init__`.

---

## `target_in_features_b.py` — `ConfirmedMomentumTilt`

**Category:** `target_in_features`

**Where:** `__init__`, the chain building `lead_return` (via `period_return.shift(-_FORWARD_HORIZON)`)
and then `strength_percentile` (a cross-sectional rank of `lead_return` divided by cohort size).
`strength_percentile` is stored in `self._confirmation` and consumed directly in `generate()` as
the "confirmation" gate (`confirmation > 0.7`).

**What it sees that it shouldn't:** `lead_return` is the forward return over the same
`_FORWARD_HORIZON`-session window the strategy is implicitly trying to profit from — it is a
shift of `period_return` into the future, not out of it. `strength_percentile` is a monotonic
transform of that forward return (a cross-sectional rank), so "confirmation > 0.7" is, in effect,
"this name is already known to be a top performer over the next few sessions" — the target
reaches the decision rule through a transform rather than as a raw duplicate column.

**Why it's easy to miss:** nothing named `next_return` or `label` appears anywhere in this file;
the leaking quantity is dressed up as a legitimate-sounding engineered feature ("confirmation
percentile", "how strongly each name leads its peers"), which is a very natural thing for a
momentum strategy to want, and the shift direction (`-_FORWARD_HORIZON`) is easy to read past
without noticing it points forward rather than backward.
