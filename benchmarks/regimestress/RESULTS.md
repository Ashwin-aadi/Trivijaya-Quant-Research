# RegimeStress v1 — RESULTS

> **This file is generated.** `RESULTS.md` is rendered from `RESULTS.template.md` by
> `scripts/build_paper_numbers.py`, which reads the frozen artifacts and substitutes every
> double-brace placeholder. Do not edit `RESULTS.md`; edit the template and regenerate. Every figure
> below therefore has a machine-checkable provenance, recorded in `paper_numbers.json` alongside
> the artifact it came from.

Reference implementation: this repository, at the commit that carries the freeze tag.
Seed 42 throughout. Local inference only.

---

## 1. Population

| Quantity | Value |
|---|---:|
| Strategies entering calibration | 185 |
| Deterministic under replicate runs and hash-seed changes | 158 |
| Excluded as nondeterministic | 27 |
| — of which audited survivors | 27 |
| Audited survivors retained | 147 |
| Standard factors among the nondeterministic exclusions | 0 |
| Worst Sharpe swing among the excluded | 1.561 |
| Failed to run at all | 2 |
| Stressed | 156 |
| In primary fragility statistics | 125 |
| Excluded as knife-edge | 31 |
| Flagged for a near-zero mean in the fragility denominator | 3 |

The determinism census was frozen **before any stress path was run**. Exclusions are reported as
counts and characterised in §6, never as silent absences.

---

## 2. Causal regime labelling

| Quantity | Value |
|---|---:|
| States, selected by BIC and then frozen permanently | 4 |
| Sessions used for state selection (all pre-development) | 1,207 |
| Selection window ends | 2019-12-31 |
| Sessions labelled | 1,233 |
| Expanding-window refits | 20 |
| Training sessions, first refit → last refit | 1,207 → 2,378 |
| Least-occupied state (sessions) | 237 |
| Most-occupied state (sessions) | 362 |
| One-quarter label revision rate | 9.5% |
| Terminal disagreement against the final refit | 21.8% |
| Dated market-structure events in the timeline | 18 |

**Independent verification of the index series the labels are fitted on.** Burn-in closes were
compared against NSE's own published index bhavcopy: 55 sessions compared,
0 beyond tolerance, worst relative difference
4.7e-8 — floating-point noise.

**Labels drift, and that is a property of causal refitting, not a defect.** A terminal disagreement
of 21.8% means roughly one session in five carries a different label
today than it did when it was labelled. Per-period summaries are therefore the primary analysis;
pooled summaries carry an explicit note that regime definitions evolve.

---

## 3. Counterfactual Regime Resampling

| Quantity | Value |
|---|---:|
| Block length (sessions), Politis & White (2004) | 11.07 |
| Bandwidth used by the selection rule | 14 |
| Calibration paths | 1,000 |
| Sessions per path | 1,232 |
| Calibration window | 2020-01-02 → 2024-12-31 |
| Target moments tested | 9 |
| Moments outside the 95% interval — conditional | 0 |
| Moments outside the 95% interval — unconditional | 1 |
| Percentile of the real value on the failing moment | 97.8 |
| Seconds to draw all calibration paths | 0.026 |

The conditional resampler reproduces every target moment. The unconditional one fails a single
long-horizon volatility-clustering statistic. **That failure is reported rather than repaired:** the
PI's Checkpoint 2.1 ruling was to keep the moment validation exactly as measured and not modify the
resampler to pass it. Conditional resampling is primary; unconditional is retained as an ablation,
and the difference between them is an empirical result rather than a design assertion.

---

## 4. The two tiers

**Tier 1** rebuilds a synthetic price panel and re-runs every strategy on it, so strategies
re-decide under counterfactual history. **Tier 2** bootstraps each strategy's *realised* return
series, which is far cheaper but cannot let a strategy respond to the path it is given.

| Quantity | Value |
|---|---:|
| Mean seconds per backtest, measured in calibration | 28.4 |
| Projected minutes for 100 paths, from that calibration | 365 |
| Tier 1 paths | 100 |
| Tier 1 cost | 124.5 CPU-hours |
| Median seconds per Tier 1 path | 4617 |
| Tier 2 paths | 1,000 |
| Tier 2 cost | 23.1 seconds |
| **Tier agreement on mean performance** (Spearman, n = 125) | **0.897** |
| **Tier agreement on fragility** (Spearman, n = 125) | **0.620** |
| Median fragility, Tier 1 | 0.360 |
| Median fragility, Tier 2 | 0.311 |

### The headline methodological result

**Tier 2 reproduces overall performance well (0.897) and fragility poorly
(0.620).** The cheap shortcut is adequate for the question *"how did this
strategy do?"* and inadequate for the question *"how much does its performance move?"* — which is
the question the benchmark exists to ask.

This is an empirical methodological finding, not a failure of Tier 2.

The divergence is **consistent with** strategy re-decision contributing materially to measured
fragility — the two tiers differ in whether a strategy may respond to the path it is given, and it
is fragility rather than performance that the cheap tier loses. **We do not claim that as a cause.**
Fragility is a variance-like quantity and therefore a noisier statistic than mean performance, so
some of the gap is expected on estimation grounds alone. Separating the two needs an experiment we
did not run: a Tier 1 variant that resamples the panel while freezing each strategy's book to its
real-history positions.

What holds regardless of the interpretation: Tier 2 cannot be substituted for Tier 1 when fragility
is the quantity of interest.

### Conditioning is close to free

| Quantity | Value |
|---|---:|
| Conditional vs unconditional, across-path fragility (Spearman) | 0.991 |
| Conditional vs unconditional, across-regime fragility (Spearman) | 0.976 |

Conditioning the bootstrap on regime label barely changes the fragility *ranking*, though it is
what makes the synthetic moments pass. Both are reported.

### The two fragility definitions are not interchangeable

Across-path and across-regime fragility agree at Spearman **0.550**
(n = 125). They measure different things and the benchmark reports both.

---

## 5. The validated shortcut

Across-regime fragility can be computed directly from a strategy's persisted real return series,
at no simulation cost. The question is whether that inexpensive computation preserves the ranking
the expensive one produces.

| Quantity | Value |
|---|---:|
| Spearman, direct computation vs bootstrap (n = 125) | **0.963** |
| Median fragility, direct | 0.618 |
| Median fragility, bootstrap | 0.551 |
| Rank convergence at 10 paths | 0.817 |
| Rank convergence at the path count Tier 1 actually ran (100) | 0.952 |
| Rank convergence at 500 paths | 0.984 |

Convergence is measured as the rank agreement between fragility computed on the first *k* bootstrap
paths and on the full set. It plateaus well before the path count used, so the expensive
experiment's path budget was not the binding constraint on ranking precision.

**The expensive experiment was used to validate the inexpensive computation rather than being
replaced by assumption.** The Tier 1 run was not skipped and then justified afterwards; it was run
in full, at 124.5 CPU-hours, and its output was then used to show that a rerun
would have reproduced a ranking already available for free.

---

## 6. What the exclusions cost

### Knife-edge strategies

A strategy is knife-edge if its Sharpe ratio is not stable under a numerically negligible change
to its inputs. 31 of 156 qualify, including
1 standard academic factor — **the rule was applied without exemption**, per
the PI ruling.

| Quantity | Excluded | Retained |
|---|---:|---:|
| Count | 31 | 125 |
| Median turnover per session | 0.887 | 0.044 |
| Median holding period (sessions) | 2.0 | 37.1 |
| Median effective holdings | 5.000 | 5.000 |

Sharpe swing among the excluded: median 0.0088, maximum
2.514, with 4 above 0.5.

**The exclusion is not random.** It removes the fast end of the corpus — twenty times the turnover
and a twentieth of the holding period — so results are conditioned on a restricted range of exactly
the characteristic that most plausibly drives fragility. Reported here rather than assumed benign.

### Exact duplicates

| Quantity | Value |
|---|---:|
| Strategies compared | 150 |
| Series too short to compare | 6 |
| Clusters of identical realised return series | 11 |
| Strategies inside a cluster | 27 |
| Largest cluster | 4 |
| Removed | 16 |
| Near-duplicate pairs (r ≥ 0.9999) detected and **not** removed | 20 |

---

## 7. Fragility prediction

Training set: 109 strategies, 26 features, 5-fold CV, seed 42.

### 7.1 The correction that removed the optimistic result

| Configuration | n | Out-of-fold R² |
|---|---:|---:|
| With duplicate rows present | 125 | +0.262 |
| Duplicates removed | 109 | +0.024 |
| **Leakage** | | **0.238** |

Identical features, identical folds, identical seed. A duplicated strategy sitting in a training
fold while its twin is scored in the test fold is worth 0.238 of R² on its own.

A second defect was found at the same time: one feature column was another under a different name,
correlating at 1.000 and driving the design's condition number to the order of 1e15. Both defects
inflated the earlier reported result. Both are described in `CORRECTIONS.md`.

### 7.2 Five models, identical folds

**Primary target — `fragility_across_paths`, raw** (Tier 1 ground truth, PI-designated)

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | -1.742 | +0.404 | 2.146 | +0.275 | 4.887 / 2.424 | -271.396 | -907.383 |
| Lasso | -2.259 | +0.424 | 2.683 | +0.220 | 5.620 / 2.424 | -343.653 | -1139.670 |
| Elastic net | -1.961 | +0.415 | 2.375 | +0.250 | 5.150 / 2.424 | -302.146 | -1008.650 |
| Random forest | -0.036 | +0.292 | 0.329 | +0.225 | 2.544 / 2.424 | -22.049 | -77.312 |
| Gradient boosting | +0.024 | +0.653 | 0.629 | +0.108 | 3.107 / 2.424 | -35.864 | -124.861 |

**`fragility_across_paths`, log1p**

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | -0.749 | +0.517 | 1.266 | +0.391 | 0.444 / 0.359 | -6.293 | -14.629 |
| Lasso | -0.270 | +0.487 | 0.757 | +0.420 | 0.389 / 0.359 | -3.187 | -7.872 |
| Elastic net | -0.473 | +0.504 | 0.977 | +0.409 | 0.412 / 0.359 | -4.498 | -10.702 |
| Random forest | +0.092 | +0.456 | 0.364 | +0.320 | 0.342 / 0.359 | -0.735 | -2.240 |
| Gradient boosting | +0.045 | +0.723 | 0.678 | +0.253 | 0.378 / 0.359 | -1.119 | -3.064 |

**`fragility_across_regimes`, raw** (the charter's definition)

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | -0.483 | +0.236 | 0.719 | +0.323 | 10.444 / 4.938 | -274.472 | -671.127 |
| Lasso | -0.770 | +0.244 | 1.014 | +0.324 | 11.847 / 4.938 | -432.688 | -1095.241 |
| Elastic net | -0.554 | +0.237 | 0.791 | +0.321 | 10.883 / 4.938 | -313.492 | -773.489 |
| Random forest | -0.067 | +0.276 | 0.343 | +0.231 | 5.694 / 4.938 | -40.280 | -90.573 |
| Gradient boosting | -0.083 | +0.511 | 0.594 | +0.286 | 6.822 / 4.938 | -48.835 | -121.306 |

**`fragility_across_regimes`, log1p — the only specification that holds together**

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | +0.039 | +0.459 | 0.420 | +0.405 | 0.409 / 0.422 | -0.595 | -1.471 |
| Lasso | +0.075 | +0.433 | 0.358 | +0.340 | 0.399 / 0.422 | -0.444 | -1.166 |
| Elastic net | +0.054 | +0.447 | 0.393 | +0.348 | 0.407 / 0.422 | -0.536 | -1.352 |
| **Random forest** | **+0.212** | +0.537 | 0.325 | +0.320 | **0.350** / 0.422 | **-0.002** | -0.539 |
| Gradient boosting | +0.198 | +0.734 | 0.537 | +0.331 | 0.378 / 0.422 | -0.135 | -0.805 |

### 7.3 Bias or variance

Every model fits its own training folds better than it generalises, and the gap is *widest* for the
model with the most capacity. On the log-regimes target the conservative random forest has both the
best out-of-fold R² (+0.212) and by far the best trimming stability
(-0.002 at drop-5, against -0.135 for gradient
boosting), while gradient boosting fits its training folds hardest
(+0.734, a gap of 0.537).

**Nothing underfits. The limitation is data variance, not model bias**, and no higher-capacity
learner was built.

### 7.4 Why the primary target is unstable

| Cause | Verdict | Evidence |
|---|---|---|
| Heavy-tailed target | **Confirmed, dominant** | Top 5 of 109 rows own 96.1% of the sum of squares (raw); skew 5.72, excess kurtosis 32.1. `log1p` reduces the top-5 share to 79.1%. |
| Multicollinearity | **Present, secondary** | Condition number 253.2; `mean_herfindahl` ↔ `mean_largest_weight_share` at \|r\| = 0.995. Destabilises the linear models; leaves the tree models alone. |
| Influential observations | **Confirmed** | Removing one row (`candidate_116`) moves out-of-fold R² by -1.859. |
| Insufficient sample | **Confirmed contributor** | Learning curve still climbing at n = 100: ρ +0.210 → +0.408 on the log-path target. Has not saturated. |
| Genuinely weak relationship | **Partially rejected** | Permutation test, 200 shuffles: p(ρ) = 0.005, 0.005, 0.005 and 0.020 across the four targets. But p(R²) reaches significance on only one — 0.005 for log-regimes, against 0.861, 0.507 and 0.572 elsewhere. |

**The conclusion.** Strategy characteristics carry statistically detectable information about the
fragility *ranking* (ρ ≈ +0.275 to +0.420, permutation
p = 0.005–0.020) and very weak information about its
*level*.

### 7.5 The full diagnostic table, all four targets

Published in full rather than for the two the argument turns on. A reader who suspects the
conclusion was reached by picking a favourable specification can check every one.

| Diagnostic | paths[raw] | paths[log1p] | regimes[raw] | regimes[log1p] |
|---|---:|---:|---:|---:|
| Skewness | 5.72 | 3.70 | 10.21 | 3.85 |
| Excess kurtosis | 32.1 | 14.7 | 102.8 | 22.1 |
| Top-5 rows' share of variance | 96.1% | 79.1% | 98.7% | 65.9% |
| Top-10 rows' share of variance | 96.4% | 85.8% | 98.8% | 76.1% |
| Best model by ρ | ridge | lasso | lasso | ridge |
| Best model by R² | gradient boosting | random forest | random forest | random forest |
| Best R² | +0.024 | +0.092 | -0.067 | +0.212 |
| Observed ρ (permutation test) | +0.275 | +0.420 | +0.324 | +0.405 |
| Null \|ρ\| 95th percentile | 0.200 | 0.209 | 0.215 | 0.259 |
| p(ρ) | 0.020 | 0.005 | 0.005 | 0.005 |
| p(R²) | 0.861 | 0.507 | 0.572 | 0.005 |
| Largest single-row ΔR² | -1.859 | +0.390 | -3.439 | -0.246 |
| Most influential row | `candidate_116` | `candidate_1005` | `candidate_490` | `candidate_274` |

**Note the pattern in the last two rows of the permutation block.** p(ρ) reaches significance on
all four specifications; p(R²) on one. The rank is learnable and the level is not, and that holds
whichever target definition and transform you choose.

Learning curves, ρ against training-set size (repeated subsampling, 109 rows
available so the largest planned size was not reachable):

| n | paths[raw] | paths[log1p] | regimes[raw] | regimes[log1p] |
|---:|---:|---:|---:|---:|
| 40 | +0.138 | +0.210 | +0.113 | +0.244 |
| 60 | +0.192 | +0.311 | +0.122 | +0.205 |
| 80 | +0.213 | +0.355 | +0.171 | +0.246 |
| 100 | +0.247 | +0.408 | +0.239 | +0.290 |

Three of the four curves rise monotonically to the largest size available. **The curve has not
saturated, and more strategies is the one intervention this diagnosis supports.** Generating them
means reopening the generator and re-running the audit and stress pipeline for the new candidates
— a separate undertaking, and left as future work.

### 7.6 Factor exposures and feature importances

Factor exposures come from **one joint regression on all factors at once**, not from separate
univariate fits. The joint design's condition number is 12.0 across
10 factors, with a maximum pairwise correlation of 0.946 — well
inside the range where joint coefficients are stable. Univariate betas are retained in the feature
table as a labelled sensitivity and are not offered to the models.

Top permutation importances on the primary target, after deduplication:

| Rank | Feature | Increase in out-of-fold MAE when shuffled |
|---:|---|---:|
| 1 | `beta_long_term_reversal_756d` | +0.408 |
| 2 | `factor_r_squared` | +0.280 |
| 3 | `trading_session_rate` | +0.187 |
| 4 | `mean_holding_period` | +0.176 |
| 5 | `beta_inverse_volatility_weighted` | +0.148 |

This ranking **changed completely** once duplicates were removed and joint betas replaced
univariate ones. The earlier ranking was substantially an artefact of duplicated rows. See
`CORRECTIONS.md`.

---

## 8. Stress narratives

12 narratives generated by `qwen2.5:7b-instruct-q4_K_M` at 5.5
seconds each, temperature 0, seed 42. The prompt contains only measured facts, and those facts are
stored beside each narrative so any claim can be audited against them.

**This is the weakest component of the benchmark and is presented as illustrative.** The narratives
are factually correct but near-vacuous: they restate the fragility measurement rather than explain
it, and none used the turnover, holding-period or concentration facts supplied to them. One
narrative describes a strategy as most reliable in a regime in which it loses money — fragility
measures consistency, not quality, and the narrative layer does not enforce that distinction.

---

## 9. Fragility distribution

Median across-regime fragility 0.618, interquartile range
[0.491, 1.377], n = 125.

Fragility is a ratio with a mean in its denominator, so strategies whose mean performance is near
zero carry a large fragility for an arithmetic reason. 3 are flagged and reported
rather than dropped.

---

## 10. Reproducing every number above

```bash
python scripts/fetch_regime_burnin.py        # index history for HMM burn-in
python scripts/cross_check_burnin.py         # against NSE's published index bhavcopy
python scripts/select_regime_states.py       # BIC over K; write-once
python scripts/build_regimes.py              # causal labels + stability diagnostics
python scripts/build_sebi_events.py          # market-structure timeline
python scripts/calibrate_crr.py              # block length + moment validation
python scripts/calibrate_tier1.py            # determinism census + timing
python scripts/run_stress_tier1.py           # the expensive experiment
python scripts/run_stress_tier2.py           # the cheap one
python scripts/compare_tiers.py              # tier and definition agreement
python scripts/is_the_rerun_needed.py        # shortcut validation
python scripts/persist_positions.py          # books, for holding period and concentration
python scripts/build_fragility.py            # both fragility definitions
python scripts/freeze_knife_edge.py          # the stability filter
python scripts/report_knife_edge.py          # what the filter costs
python scripts/deduplicate_corpus.py         # exact and near-duplicate detection
python scripts/build_characteristics.py      # features, with joint factor betas
python scripts/train_fragility_predictor.py  # the predictor, honestly scored
python scripts/diagnose_predictor.py         # five models, five causes
python scripts/generate_narratives.py        # the illustrative layer
python scripts/build_paper_numbers.py        # regenerates this file and the paper's macros
```

`scripts/run_stress_tier1.py` is the only expensive step (124.5 CPU-hours). Every
other step completes in minutes.
