# Ground truth: `tests/fixtures/locked/honest/`

Twelve negative-control strategies. Every one is genuinely leak-free: each reads only from the
`MarketView` handed to `generate`, each constructor takes only scalar settings (window lengths,
thresholds, counts) and never a frame, and no code path reaches a session on or after `as_of`.
These exist to measure the auditor's false-positive rate, so the case for honesty is argued
explicitly below for each file.

---

## Naming traps (1–6)

These files are completely honest strategies that deliberately use vocabulary which, read as
words alone, sounds like a leakage class the static auditor screens for. The claim under test is
that the auditor judges data-flow structure, not identifier names.

### `trap_1_target_and_label.py`
**Suggestive names used:** `label`, `target_weight`.
Its only data source is `view.closes(self.lookback)`, a pivot already bounded to sessions
strictly before the decision date; `label` is a per-symbol trailing-return score computed from
the first and last rows of that bounded frame, never a supervised-learning target pulled from a
future session. `target_weight` is nothing but the `Signal.weights` dict every strategy in this
codebase produces — "target" describes the portfolio being aimed for going forward, not a
leaked outcome variable.

### `trap_2_final_and_end.py`
**Suggestive names used:** `final_allocation`, `end_state`.
`end_state` is bound to `closes.tail(1).row(0, named=True)` — the newest row inside a frame that
`view.closes` has already clipped to strictly-before-`as_of` sessions. "End" here means "the end
of the visible window," not "the end of some later, unseen period." `final_allocation` is the
ordinary weights dict returned in the `Signal`; nothing about the word "final" implies it was
computed with information from beyond the decision date.

### `trap_3_full_and_entire.py`
**Suggestive names used:** `full_window`, `entire_range`.
Both names are bound to `view.closes()` called with no lookback argument, i.e. every session
`MarketView` has revealed so far — which, by construction, is still hard-truncated at `as_of`.
"Full"/"entire" describes the *entire visible* history, not a full-sample statistic computed
across train and test together. The `.mean()` taken over that frame is exactly the trailing,
point-in-time statistic the engine sanctions via `view.history()`/`view.closes()`.

### `trap_4_today_and_latest.py`
**Suggestive names used:** `today`, `latest_level`.
`today` is bound to `view.history(1)["session_date"].max()` — the newest session already printed
— and is explicitly never bound to `view.as_of`, which is the session the engine is about to fill
and which `MarketView` refuses to expose. `latest_level` is the last row of a `closes` frame that
is itself bounded to the same already-past window. The name "today" is a deliberately confusing
label for what is, mechanically, "the most recent already-closed session," which is also exactly
the timestamp used to stamp the `Signal`.

### `trap_5_panel_and_snapshot.py`
**Suggestive names used:** `panel`, `snapshot`.
`panel` is bound to `view.history(self.breakout_window)`, the caller-facing, already-truncated
frame `MarketView` hands out — not the engine's internal, unrestricted table. `snapshot` is
`view.latest_close()`, the most recent close per symbol that has actually printed. Neither name
implies access to any object other than the sanctioned `view` accessors.

### `trap_6_survivor_and_forward.py`
**Suggestive names used:** `survivor_count`, `forward_weight`.
`survivor_count` counts, within the already point-in-time universe `view.symbols` supplied by the
engine, how many symbols have enough trailing observations in `view.closes(lookback)` to be
scored — a data-sufficiency screen applied to today's universe, not a selection based on which
names remained in an index at the end of the study period. `forward_weight` is simply the weight
carried into the next session, i.e. `Signal.weights`.

---

## Plain honest strategies (7–12)

Ordinary, defensible, probably-unprofitable constructions of the kind a careful researcher
writes. Several deliberately include constructions that superficially resemble leakage —
a trailing `.mean()`, a positive `.shift(1)`, a rolling window, a cross-sectional sort — to check
that the auditor does not flag legitimate, backward-only uses of those same primitives.

### `plain_1_ma_crossover.py`
Only data source: `view.closes(self.long_window)`. The "short" average is `series.tail(n).mean()`
taken from *inside* that already-bounded long window, so it ends at the same last visible session
as the long average and starts no earlier than the long window's own start — a trailing tail of a
trailing window, never centred and never reaching past the decision date.

### `plain_2_volatility_tilt.py`
Only data source: `view.closes(self.vol_window + 1)`. Daily returns are built with a single,
positive `price.shift(1)`, i.e. each session is compared to the one immediately before it. A
positive shift is a lag, not a lead: it can only look backward. Volatility is `std()` over that
lagged, bounded return series.

### `plain_3_cross_sectional_rank.py`
Only data source: `view.history(self.lookback)`. The per-symbol trailing return is computed with
a `group_by("symbol")` aggregation using `.first()`/`.last()` after sorting by
`["symbol", "session_date"]` — both are polars aggregations over rows that already belong to the
bounded frame, so "first" and "last" mean "oldest and newest visible session in this window,"
never a peek past it. Sorting the resulting cross-section and taking `.head(top_n)` is an
ordinary rank-and-select, not a lookahead device.

### `plain_4_lagged_momentum.py`
Only data source: `view.closes(self.formation_window)`. Returns are built with `price.shift(1)`
(a positive, backward-only lag), and the score additionally drops the most recent `skip`
observations from the *end* of that trailing series via a forward slice
`daily_return[1 : daily_return.len() - self.skip]` — both operations move the window further into
the past, never closer to or past the decision boundary.

### `plain_5_liquidity_screened_equal_weight.py`
Only data source: `view.history(self.adv_window)`, intersected with `view.symbols` (the
point-in-time investable universe already handed to the strategy). Average daily volume is a
`group_by("symbol").agg(pl.col("volume").mean())` over that bounded frame — a trailing liquidity
statistic, not a forward-looking one, and the candidate list is never widened past what `view`
already restricted it to.

### `plain_6_trailing_reversion.py`
Only data source: `view.closes(self.lookback)`. Trailing mean and standard deviation are computed
over that bounded frame, and the most recent visible close is compared against a statistic that
legitimately includes it — the same construction a Bollinger-style band uses. Nothing in the
z-score reaches beyond the last row `view.closes` returned.

---

## Self-check performed

Every one of the twelve files above was re-read in full, file by file, and checked against this
list before this document was written:

- Constructor accepts only scalars (`int`, `float`) — never a `pl.DataFrame` or any object holding
  the panel.
- `generate` reads nothing except `view.history(...)`, `view.closes(...)`, `view.latest_close()`,
  and `view.symbols` — no other data source, no closed-over module state.
- No negative shift (`.shift(-n)`) anywhere; the only shifts used are `.shift(1)`, a backward lag.
- No forward slice and no reversed slice; the one explicit slice
  (`daily_return[1 : daily_return.len() - self.skip]` in `plain_4`) has non-negative, increasing
  bounds strictly inside an already-bounded series.
- No centred rolling window; every average/std is either a whole-frame `.mean()`/`.std()` over an
  already-truncated frame, or a `.tail(n)` of one — never `center=True`.
- No backward fill (`fill_null(strategy="backward")` or equivalent) anywhere; nulls are dropped or
  cause the symbol to be skipped.
- No module-level data (no top-level `pl.DataFrame`, no cached constants derived from data).
- No statistic computed over anything wider than what `view.history()`/`view.closes()` returned
  for that call — lookback arguments are always passed through, and where `view.closes()` is
  called with no lookback (`trap_3`), the width is still the full *visible* history, which
  `MarketView` truncates at construction.
- Every `Signal.information_available_at` is derived from `view.history(1)["session_date"].max()`
  (or an equivalent bounded lookup), guarded to fall back to `date.min` when no history is visible
  yet, and `view.as_of` is never referenced anywhere in any of the twelve files.

All twelve files were also confirmed to parse (`ast.parse`) and pass
`ruff check tests/fixtures/locked/honest/` with zero findings.
