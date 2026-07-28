# Statistical deflation: what the trial count and the evaluation window cost an apparent edge

Frozen record of every number the statistical auditor (`src/audit/stat.py`) contributes to
Checkpoint 1.3. Nothing here is fitted or tuned. Each figure is the output of running existing code
on real data, and every one is reported with the sample size it rests on.

Reproduce the whole document with:

```
python scripts/run_stat_audit.py
```

| Run property | Value |
|---|---|
| Wall clock | 547.7 s |
| Manifest | `runs/20260725T185627_428382/manifest.json` |
| Seed | 42 |
| Price panel | `data/processed/prices_adjusted.parquet` (2,287,765 rows, 2019-06-25 to 2024-12-31) |
| Universe | `data/processed/universe.parquet` |
| Calendar | `data/raw/calendar_cnx100.parquet` |

---

## A note on units, before any number is read

`src/audit/stat.py` works in **per-observation** Sharpe ratios — mean divided by sample standard
deviation at daily frequency. Its docstring is explicit that passing an annualised Sharpe alongside
a daily observation count inflates the answer badly, so every conversion below is spelled out:
`SR_daily = SR_annual / sqrt(252)` and `V_daily = V_annual / 252`.

Kurtosis throughout is **non-excess** — 3.0 for a normal distribution. The module rejects a value
below 1.0 precisely because excess kurtosis passed by mistake would shrink the standard error and
flatter the result.

**One annualised Sharpe convention, project-wide.** `src/eval/metrics.py` annualises the
*arithmetic* mean return by 252 sessions and scales volatility by `sqrt(252)`, which is the form
the Deflated Sharpe is defined on. A Sharpe quoted anywhere in this repository can therefore be
fed to the DSR directly.

This was not always so. The module previously annualised geometrically over 250 sessions, and on
the live control below the two conventions differed by 12.9% over one window and 2.9% the other
way. Neither was wrong — they answer different questions — but carrying both meant a figure could
reach the deflation machinery under the wrong definition, and a 12.9% overstatement of a Sharpe
becomes a much larger overstatement of significance once deflated. The conventions were unified on
the arithmetic form rather than reconciled at each call site, because a conversion applied at a
boundary is a step someone eventually forgets. Cumulative growth is still available, under the
separate and unambiguous name `total_return`.

Consequence for figures published before the change: the live control's `metrics.py` Sharpes move
from 2.8529 to 2.5265 (2023) and from 0.9448 to 0.9728 (full window). The deflation results are
unaffected, because they were computed on the arithmetic figures throughout.

---

## 1. The worked example

A hypothetical strategy showing an **annualised Sharpe of 2.0 over ten years of daily data**
(2,520 observations), found after **200 trials**, with modestly non-normal returns
(skewness -0.5, non-excess kurtosis 5.0).

| Quantity | Value | Sample size |
|---|---|---|
| Observed Sharpe, annualised | 2.0 | 2,520 daily observations |
| Observed Sharpe, per observation | 0.12598816 | 2,520 |
| Standard error of the Sharpe estimate, per observation | 0.02069521 | 2,520 |
| Probabilistic Sharpe Ratio against zero (no deflation) | 1.00000000 (to 8 dp) | 2,520 |

The PSR says the sample is long enough that the true Sharpe being above zero is not in doubt. All
of the doubt introduced below comes from the trial count, none of it from the sample length.

The deflation depends on one further input, `V` — the variance of Sharpe ratios *across the 200
trials*. This is the dispersion of the search, not of the returns, and it is the term most often
omitted. Two values bracket a plausible range and both are reported, because the sensitivity to `V`
is the honest caveat on the whole method:

| Annual `V` | Daily `V` | Luck threshold `E[max SR]`, daily | Luck threshold, annualised | Observed − threshold | **DSR** |
|---|---|---|---|---|---|
| 0.25 (trial Sharpe sd 0.5) | 0.00099206 | 0.08710582 | 1.382762 | +0.03888234 | **0.96986468** |
| 1.00 (trial Sharpe sd 1.0) | 0.00396825 | 0.17421163 | 2.765524 | −0.04822347 | **0.00989844** |

### Why the number drops

Deflation is not a haircut applied to the Sharpe. It moves the *benchmark*. The PSR asks "is the
true Sharpe above zero"; the DSR asks "is it above the Sharpe the luckiest of 200 skill-free
strategies would be expected to show anyway".

At a trial-Sharpe dispersion of 1.0 annualised, that threshold is **2.77 annualised** — higher than
the 2.0 actually observed. The excess is negative, so the strategy is *worse* than what searching
200 times through a pool that varied is expected to throw up by chance, and the DSR collapses to
0.0099. At a dispersion of 0.5 the threshold is only 1.38, the excess is positive, and the DSR stays
at 0.97.

Same strategy, same raw Sharpe, same trial count. **The DSR is not a property of the strategy
alone.** Halving the assumed dispersion of the search moves the answer from "almost certainly
noise" to "almost certainly real". Any DSR quoted in this project must carry the `V` it was computed
against, and that `V` must be measured from the corpus rather than assumed.

These two figures are frozen in `tests/audit/test_stat.py::test_worked_example_for_the_checkpoint_report`
so they cannot drift silently under a refactor.

---

## 2. Sensitivity to search intensity

The same strategy — annualised Sharpe 2.0, 2,520 observations, skew -0.5, kurtosis 5.0 — at rising
trial counts. Everything else is held fixed; only `N` moves.

| Trials `N` | Threshold, annualised (`V`=0.25) | **DSR** (`V`=0.25) | Threshold, annualised (`V`=1.00) | **DSR** (`V`=1.00) |
|---|---|---|---|---|
| 1 | 0.000000 | 1.00000000 | 0.000000 | 1.00000000 |
| 10 | 0.787299 | 0.99988846 | 1.574598 | 0.90231895 |
| 50 | 1.138152 | 0.99564686 | 2.276303 | 0.20016334 |
| 100 | 1.265301 | 0.98733545 | 2.530603 | 0.05314451 |
| 200 | 1.382762 | 0.96986468 | 2.765524 | 0.00989844 |
| 500 | 1.526264 | 0.92534936 | 3.052528 | 0.00067817 |
| 1000 | 1.627561 | 0.87153275 | 3.255122 | 0.00006660 |

Sample size is 2,520 daily observations at every row.

At `N`=1 the deflation is exactly nothing: a single attempt is not a search, so there is no
selection to correct for and the DSR equals the undeflated PSR. From there the threshold rises
roughly with the square root of the log of `N` — the curve is steep early and flattens, which is why
the difference between trying 1 strategy and trying 50 matters far more than the difference between
500 and 1000.

The two columns fall at very different rates. Under a narrow search the same Sharpe survives a
thousand trials with a DSR of 0.87; under a wide one it is gone by fifty. Trial count and search
dispersion are not interchangeable and neither alone is sufficient to state a deflated result.

---

## 3. Live positive control: `EqualWeightUniverse`

`tests/fixtures/clean/equal_weight_universe.py` holds every name in the point-in-time universe at
equal weight. It is not a leaky fixture, it expresses no view, and it has no parameters to fit. It
is used here because it demonstrates the failure mode without any cheating being involved: a real
strategy on real data, quoted over a favourable one-year window, looks excellent.

Both figures were reproduced by running `BacktestEngine` over the panel. No costs are applied
(`cost_per_turnover` defaults to 0.0; the Indian cost model is a later phase).

| Window | Sessions | Sharpe (`metrics.py`) | Sharpe (arithmetic, ×√252) | Sharpe per observation | Skew | Kurtosis (non-excess) |
|---|---|---|---|---|---|---|
| 2023-01-01 to 2023-12-31 | 244 | **2.8529** | 2.5265 | 0.159153 | −0.8166 | 4.5153 |
| 2020-01-01 to 2024-12-31 | 1,232 | **0.9448** | 0.9728 | 0.061280 | −1.6545 | 15.6393 |

Skewness and kurtosis are computed from the realised return series with `scipy.stats`
(`skew(bias=False)`, `kurtosis(fisher=False, bias=False)`), not assumed.

The first honest observation needs no statistics at all: **the same strategy, on the same data,
with no parameter changed, shows a Sharpe of 2.85 on one window and 0.94 on the window containing
it.** The 2023 figure is not wrong. It is a true statement about 244 sessions, and it is
approximately meaningless as a statement about the strategy.

The full-window series is also far worse behaved — skew −1.65 against −0.82, kurtosis 15.6 against
4.5. The one-year window excludes the 2020 drawdown, so it is quoting the calm part of the sample.

### Deflating the 2023 figure

`V` is measured, not assumed: all 30 clean fixtures were run over the identical 2023 window and the
sample variance of their per-observation Sharpes taken.

| Quantity | Value | Sample size |
|---|---|---|
| Trial-Sharpe variance `V`, per observation | 0.0088647182 | 30 fixtures |
| Same, annualised | 2.233909 | 30 fixtures |
| Trial-Sharpe sd, annualised | 1.4946 | 30 fixtures |
| PSR of the 2023 series against zero, before deflation | 0.989591 | 244 sessions |

The corpus is genuinely dispersed — an annualised trial-Sharpe standard deviation of 1.49 across
thirty simple, honest rules over a single year. That dispersion is what sets the bar.

| Trials `N` | Luck threshold, annualised | Observed − threshold (per obs) | **DSR** |
|---|---|---|---|
| 1 | 0.000000 | +0.15915287 | 0.98959099 |
| 10 | 2.353437 | +0.01090030 | 0.56288885 |
| **30** (the corpus actually run) | **3.098948** | **−0.03606247** | **0.30023988** |
| 100 | 3.782307 | −0.07911005 | 0.12530657 |
| 200 | 4.133426 | −0.10122848 | 0.07077108 |
| 500 | 4.562389 | −0.12825063 | 0.03126656 |

Sample size is 244 sessions at every row.

Read left to right: quoted alone, the 2023 Sharpe of 2.85 carries a PSR of 0.99 and looks
conclusive. Stated honestly — that it is the best-looking window of a strategy drawn from a search
over thirty candidates whose Sharpes had an annualised spread of 1.49 — the bar it has to clear is
an annualised **3.10**, which it does not clear, and the DSR falls to **0.30**. By 100 trials it is
0.13, and by 500 it is 0.03.

Two of the three ingredients of that collapse are things a reader is not usually shown: the window
was chosen after the fact, and the trial count was not one.

---

## 4. Probability of Backtest Overfitting

A different question from the DSR. CSCV cuts the history into contiguous blocks, enumerates every
balanced split into training and testing halves, picks the best strategy in-sample, and records how
often that pick lands at or below the median out-of-sample.

Corpus: the **first twelve clean fixtures in sorted filename order**, run over the full development
window. The selection rule was fixed before any of them was run, so the corpus cannot have been
chosen on its answer.

| Fixture | Sharpe (`metrics.py`) | Sharpe per observation | Zero-return session fraction |
|---|---|---|---|
| bollinger_reversion | −1.2407 | −0.086442 | 0.3222 |
| breakout_20d | −1.2086 | −0.081593 | 0.0357 |
| cross_sectional_zscore | 0.6709 | 0.047289 | 0.0000 |
| donchian_channel | 0.5610 | 0.039622 | 0.0162 |
| dual_momentum_21_126 | 0.7760 | 0.052080 | 0.0081 |
| equal_risk_contribution_pairs | 0.9680 | 0.062373 | 0.0000 |
| equal_weight_top_liquidity | 0.8728 | 0.057557 | 0.0000 |
| equal_weight_universe | 0.9448 | 0.061280 | 0.0008 |
| gap_fade | −1.0967 | −0.077005 | 0.5227 |
| high_volatility | 0.4689 | 0.038640 | 0.0000 |
| inverse_volatility_weighted | 0.7970 | 0.051898 | 0.0000 |
| long_term_reversal_756d | 0.7049 | 0.046885 | 0.5114 |

All twelve series cover the same 1,232 sessions (2020-01-01 to 2024-12-31).

| Quantity | Value |
|---|---|
| Matrix | 1,232 sessions × 12 strategies |
| Splits | 16 |
| **Partitions enumerated** | **12,870** |
| **PBO** | **0.588423** |

**0.5 is what pure noise gives.** At 0.59 the in-sample winner lands below the out-of-sample median
slightly more often than a coin flip — selecting the best backtest in this corpus is not merely
uninformative, it is marginally worse than picking at random. That is the expected result for a
corpus of simple rules with no persistent edge among them, and it is reported as such.

The 0.09 above 0.5 should not be over-read. `tests/audit/test_stat.py` records deliberately that a
single PBO estimate on pure noise ranges over most of the unit interval across replications, so the
distance from 0.5 here is well inside the statistic's own error bar.

### Sensitivities

Split count, same corpus:

| Splits | Partitions | PBO |
|---|---|---|
| 4 | 6 | 0.666667 |
| 8 | 70 | 0.642857 |
| 12 | 924 | 0.575758 |
| 16 | 12,870 | 0.588423 |

Two fixtures hold nothing on more than half of all sessions. This is an artefact of the data start
rather than strategy behaviour — the panel begins 2019-06-25, so a 756-session lookback has no
history to act on until 2022. The threshold (50% zero-return sessions) was fixed before the corpus
was run and is *not* applied to the headline figure:

| Corpus | Strategies | Splits | PBO |
|---|---|---|---|
| First twelve (headline) | 12 | 16 | 0.588423 |
| Dropping `gap_fade`, `long_term_reversal_756d` | 10 | 16 | 0.530614 |

Every variant sits at or above 0.5. The conclusion — that in-sample ranking within this corpus
carries no usable out-of-sample information — does not depend on which variant is read.

---

## 5. Trial counter

The DSR is a function of `N`. `N` is the one input nobody can audit from the output: a deflated
Sharpe computed against an understated trial count looks exactly as respectable as an honest one.
So `N` is not remembered and passed in — it is read off an append-only, hash-chained ledger in
which each entry's digest covers the previous one.

Demonstration below runs against a temporary path. **The project's real ledger was not opened.**

Five trials recorded, including three failures:

| Seq | Name | Outcome |
|---|---|---|
| 1 | candidate_001 | evaluated |
| 2 | candidate_002 | syntax_error |
| 3 | candidate_003 | runtime_error |
| 4 | candidate_004 | evaluated |
| 5 | candidate_005 | syntax_error |

```
count()  = 5      failures included, which is the point
verify() = 5      entries verified
```

Failures consumed search effort exactly as successes did. A ledger recording only the two
strategies that ran to completion would report `N`=2 instead of `N`=5, which lowers the luck
threshold, which raises the deflated Sharpe of every survivor — a systematic bias in the flattering
direction, invisible from the output.

Line 3 was then hand-edited to relabel the runtime error as an evaluation, exactly as somebody with
a text editor would, leaving the recorded hash untouched:

```
count()  = 5      counting is not verification
verify() raised TamperError:
  contents hash to 96528156ad3065e1... but the line records 976632bbacec68dc...,
  so this entry was edited
```

Note that `count()` still returns 5. Counting lines is not verification, and any `N` quoted in a
report must come from a ledger that `verify()` has just passed.

The two digests above will differ on every re-run and are not reproducible figures: each entry's
UTC timestamp is part of the hashed payload. What reproduces is the behaviour — `verify()` raising
`TamperError` and naming the line.

**What this does and does not protect against.** It detects edits, deletions, reorderings and
insertions made by anything that cannot rewrite the whole file. It does not defend against a process
with full write access that recomputes the entire chain, nor against deletion of the file outright.
The defence there is location: `default_counter_path` places the ledger under
`data/processed/trial_ledger.jsonl`, outside the working area of whatever process proposes
strategies, and that process is expected to hold no write access to it.

---

## Summary of headline figures

| Figure | Value | Sample size |
|---|---|---|
| Worked example: raw annualised Sharpe | 2.0 | 2,520 |
| Worked example: DSR at `N`=200, `V`=0.25 annualised | 0.96986468 | 2,520 |
| Worked example: DSR at `N`=200, `V`=1.00 annualised | 0.00989844 | 2,520 |
| Live control: Sharpe over 2023 alone | 2.8529 | 244 sessions |
| Live control: Sharpe over 2020-2024 | 0.9448 | 1,232 sessions |
| Live control: PSR of the 2023 figure, undeflated | 0.989591 | 244 sessions |
| Live control: DSR of the 2023 figure at the measured `N`=30, `V`=2.234 annualised | 0.30023988 | 244 sessions |
| PBO, twelve clean fixtures, 12,870 partitions | 0.588423 | 1,232 sessions × 12 |

## Known limitations of these numbers

- **No transaction costs.** Every backtest above runs at `cost_per_turnover = 0.0`. The Indian cost
  model is a later phase. Costs move Sharpes down, so the deflated figures here are, if anything,
  optimistic.
- **`V` measured from thirty fixtures.** The trial-Sharpe variance driving the live control's
  deflation is a sample variance over 30 observations and is itself noisy. The DSR is highly
  sensitive to it, as section 1 shows.
- **One PBO estimate is a wide interval.** The 0.588 figure should be read as "at or slightly above
  the noise level of 0.5", not as a precise quantity.
- **The corpus is not a random sample of strategies.** These are 30 hand-written simple rules
  selected to be honest, not to be representative. Statistics computed across them describe this
  corpus.
- **Two annualisation conventions coexist in the repository** (`metrics.py` geometric over 250
  sessions; the DSR arithmetic over 252). This is unreconciled and is flagged rather than fixed,
  since changing `metrics.py` is outside this measurement's scope.
