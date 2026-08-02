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
| Strategies entering calibration | {{rsNCalibrated}} |
| Deterministic under replicate runs and hash-seed changes | {{rsNDeterministic}} |
| Excluded as nondeterministic | {{rsNNondeterministic}} |
| — of which audited survivors | {{rsNExcludedNondeterministic}} |
| Audited survivors retained | {{rsNRetainedSurvivors}} |
| Standard factors among the nondeterministic exclusions | {{rsNNondeterministicFactors}} |
| Worst Sharpe swing among the excluded | {{rsWorstNondeterministicSwing}} |
| Failed to run at all | {{rsNRuntimeFailures}} |
| Stressed | {{rsNStressed}} |
| In primary fragility statistics | {{rsNPrimary}} |
| Excluded as knife-edge | {{rsNKnifeEdge}} |
| Flagged for a near-zero mean in the fragility denominator | {{rsNNearZeroMean}} |

The determinism census was frozen **before any stress path was run**. Exclusions are reported as
counts and characterised in §6, never as silent absences.

---

## 2. Causal regime labelling

| Quantity | Value |
|---|---:|
| States, selected by BIC and then frozen permanently | {{rsRegimeK}} |
| Sessions used for state selection (all pre-development) | {{rsRegimeSelectionSessions}} |
| Selection window ends | {{rsRegimeSelectionEnd}} |
| Sessions labelled | {{rsNLabelledSessions}} |
| Expanding-window refits | {{rsNRegimeRefits}} |
| Training sessions, first refit → last refit | {{rsRegimeTrainingSessionsFirst}} → {{rsRegimeTrainingSessionsLast}} |
| Least-occupied state (sessions) | {{rsRegimeOccupancyMin}} |
| Most-occupied state (sessions) | {{rsRegimeOccupancyMax}} |
| One-quarter label revision rate | {{rsRegimeRevisionRate}}% |
| Terminal disagreement against the final refit | {{rsRegimeTerminalDisagreement}}% |
| Dated market-structure events in the timeline | {{rsNSebiEvents}} |

**Independent verification of the index series the labels are fitted on.** Burn-in closes were
compared against NSE's own published index bhavcopy: {{rsBurninCompared}} sessions compared,
{{rsBurninBeyondTolerance}} beyond tolerance, worst relative difference
{{rsBurninWorstRelative}} — floating-point noise.

**Labels drift, and that is a property of causal refitting, not a defect.** A terminal disagreement
of {{rsRegimeTerminalDisagreement}}% means roughly one session in five carries a different label
today than it did when it was labelled. Per-period summaries are therefore the primary analysis;
pooled summaries carry an explicit note that regime definitions evolve.

---

## 3. Counterfactual Regime Resampling

| Quantity | Value |
|---|---:|
| Block length (sessions), Politis & White (2004) | {{rsBlockLength}} |
| Bandwidth used by the selection rule | {{rsBlockBandwidth}} |
| Calibration paths | {{rsCrrPaths}} |
| Sessions per path | {{rsCrrSessions}} |
| Calibration window | {{rsCrrWindowStart}} → {{rsCrrWindowEnd}} |
| Target moments tested | {{rsNMoments}} |
| Moments outside the 95% interval — conditional | {{rsMomentsFailedConditional}} |
| Moments outside the 95% interval — unconditional | {{rsMomentsFailedUnconditional}} |
| Percentile of the real value on the failing moment | {{rsFailingMomentPercentile}} |
| Seconds to draw all calibration paths | {{rsCrrDrawSeconds}} |

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
| Mean seconds per backtest, measured in calibration | {{rsCalibrationSecondsPerBacktest}} |
| Projected minutes for 100 paths, from that calibration | {{rsCalibrationProjectedMinutes}} |
| Tier 1 paths | {{rsTierOnePaths}} |
| Tier 1 cost | {{rsTierOneCpuHours}} CPU-hours |
| Median seconds per Tier 1 path | {{rsTierOneSecondsPerPath}} |
| Tier 2 paths | {{rsTierTwoPaths}} |
| Tier 2 cost | {{rsTierTwoSeconds}} seconds |
| **Tier agreement on mean performance** (Spearman, n = {{rsTierAgreementN}}) | **{{rsTierAgreementMeanSharpe}}** |
| **Tier agreement on fragility** (Spearman, n = {{rsTierAgreementN}}) | **{{rsTierAgreementFragility}}** |
| Median fragility, Tier 1 | {{rsTierMedianOne}} |
| Median fragility, Tier 2 | {{rsTierMedianTwo}} |

### The headline methodological result

**Tier 2 reproduces overall performance well ({{rsTierAgreementMeanSharpe}}) and fragility poorly
({{rsTierAgreementFragility}}).** The cheap shortcut is adequate for the question *"how did this
strategy do?"* and inadequate for the question *"how much does its performance move?"* — which is
the question the benchmark exists to ask.

This is an empirical methodological finding, not a failure of Tier 2. It says something specific:
fragility is generated substantially by strategies *responding* to the path they are given, and a
resampling scheme that holds the decisions fixed discards the mechanism.

### Conditioning is close to free

| Quantity | Value |
|---|---:|
| Conditional vs unconditional, across-path fragility (Spearman) | {{rsConditioningEffectPaths}} |
| Conditional vs unconditional, across-regime fragility (Spearman) | {{rsConditioningEffectRegimes}} |

Conditioning the bootstrap on regime label barely changes the fragility *ranking*, though it is
what makes the synthetic moments pass. Both are reported.

### The two fragility definitions are not interchangeable

Across-path and across-regime fragility agree at Spearman **{{rsDefinitionAgreement}}**
(n = {{rsTierAgreementN}}). They measure different things and the benchmark reports both.

---

## 5. The validated shortcut

Across-regime fragility can be computed directly from a strategy's persisted real return series,
at no simulation cost. The question is whether that inexpensive computation preserves the ranking
the expensive one produces.

| Quantity | Value |
|---|---:|
| Spearman, direct computation vs bootstrap (n = {{rsShortcutN}}) | **{{rsShortcutSpearman}}** |
| Median fragility, direct | {{rsShortcutMedianReal}} |
| Median fragility, bootstrap | {{rsShortcutMedianBootstrap}} |
| Rank convergence at {{rsConvergenceLowPaths}} paths | {{rsConvergenceLow}} |
| Rank convergence at the path count Tier 1 actually ran ({{rsTierOnePaths}}) | {{rsConvergenceAtHundred}} |
| Rank convergence at {{rsConvergenceHighPaths}} paths | {{rsConvergenceHigh}} |

Convergence is measured as the rank agreement between fragility computed on the first *k* bootstrap
paths and on the full set. It plateaus well before the path count used, so the expensive
experiment's path budget was not the binding constraint on ranking precision.

**The expensive experiment was used to validate the inexpensive computation rather than being
replaced by assumption.** The Tier 1 run was not skipped and then justified afterwards; it was run
in full, at {{rsTierOneCpuHours}} CPU-hours, and its output was then used to show that a rerun
would have reproduced a ranking already available for free.

---

## 6. What the exclusions cost

### Knife-edge strategies

A strategy is knife-edge if its Sharpe ratio is not stable under a numerically negligible change
to its inputs. {{rsKnifeEdgeExcluded}} of {{rsNStressed}} qualify, including
{{rsKnifeEdgeFactors}} standard academic factor — **the rule was applied without exemption**, per
the PI ruling.

| Quantity | Excluded | Retained |
|---|---:|---:|
| Count | {{rsKnifeEdgeExcluded}} | {{rsKnifeEdgeRetained}} |
| Median turnover per session | {{rsKnifeEdgeTurnoverExcluded}} | {{rsKnifeEdgeTurnoverRetained}} |
| Median holding period (sessions) | {{rsKnifeEdgeHoldingExcluded}} | {{rsKnifeEdgeHoldingRetained}} |
| Median effective holdings | {{rsKnifeEdgeHoldingsExcluded}} | {{rsKnifeEdgeHoldingsRetained}} |

Sharpe swing among the excluded: median {{rsKnifeEdgeSwingMedian}}, maximum
{{rsKnifeEdgeSwingMax}}, with {{rsKnifeEdgeSwingAboveHalf}} above 0.5.

**The exclusion is not random.** It removes the fast end of the corpus — twenty times the turnover
and a twentieth of the holding period — so results are conditioned on a restricted range of exactly
the characteristic that most plausibly drives fragility. Reported here rather than assumed benign.

### Exact duplicates

| Quantity | Value |
|---|---:|
| Strategies compared | {{rsDuplicateCompared}} |
| Series too short to compare | {{rsDuplicateUncompared}} |
| Clusters of identical realised return series | {{rsDuplicateClusters}} |
| Strategies inside a cluster | {{rsDuplicateMembers}} |
| Largest cluster | {{rsLargestDuplicateCluster}} |
| Removed | {{rsDuplicateRemoved}} |
| Near-duplicate pairs (r ≥ {{rsNearDuplicateThreshold}}) detected and **not** removed | {{rsNearDuplicatePairs}} |

---

## 7. Fragility prediction

Training set: {{rsPredictorRows}} strategies, {{rsPredictorFeatures}} features, 5-fold CV, seed 42.

### 7.1 The correction that removed the optimistic result

| Configuration | n | Out-of-fold R² |
|---|---:|---:|
| With duplicate rows present | {{rsLeakageNWith}} | {{rsLeakageWithDuplicates}} |
| Duplicates removed | {{rsLeakageNWithout}} | {{rsLeakageWithoutDuplicates}} |
| **Leakage** | | **{{rsLeakageDelta}}** |

Identical features, identical folds, identical seed. A duplicated strategy sitting in a training
fold while its twin is scored in the test fold is worth {{rsLeakageDelta}} of R² on its own.

A second defect was found at the same time: one feature column was another under a different name,
correlating at 1.000 and driving the design's condition number to the order of 1e15. Both defects
inflated the earlier reported result. Both are described in `CORRECTIONS.md`.

### 7.2 Five models, identical folds

**Primary target — `fragility_across_paths`, raw** (Tier 1 ground truth, PI-designated)

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | {{rsPathsRawRidgeRsquared}} | {{rsPathsRawRidgeTrainRsquared}} | {{rsPathsRawRidgeGap}} | {{rsPathsRawRidgeSpearman}} | {{rsPathsRawRidgeMae}} / {{rsPathsRawBaselineMae}} | {{rsPathsRawRidgeDropFive}} | {{rsPathsRawRidgeDropTen}} |
| Lasso | {{rsPathsRawLassoRsquared}} | {{rsPathsRawLassoTrainRsquared}} | {{rsPathsRawLassoGap}} | {{rsPathsRawLassoSpearman}} | {{rsPathsRawLassoMae}} / {{rsPathsRawBaselineMae}} | {{rsPathsRawLassoDropFive}} | {{rsPathsRawLassoDropTen}} |
| Elastic net | {{rsPathsRawElasticNetRsquared}} | {{rsPathsRawElasticNetTrainRsquared}} | {{rsPathsRawElasticNetGap}} | {{rsPathsRawElasticNetSpearman}} | {{rsPathsRawElasticNetMae}} / {{rsPathsRawBaselineMae}} | {{rsPathsRawElasticNetDropFive}} | {{rsPathsRawElasticNetDropTen}} |
| Random forest | {{rsPathsRawForestRsquared}} | {{rsPathsRawForestTrainRsquared}} | {{rsPathsRawForestGap}} | {{rsPathsRawForestSpearman}} | {{rsPathsRawForestMae}} / {{rsPathsRawBaselineMae}} | {{rsPathsRawForestDropFive}} | {{rsPathsRawForestDropTen}} |
| Gradient boosting | {{rsPathsRawBoostingRsquared}} | {{rsPathsRawBoostingTrainRsquared}} | {{rsPathsRawBoostingGap}} | {{rsPathsRawBoostingSpearman}} | {{rsPathsRawBoostingMae}} / {{rsPathsRawBaselineMae}} | {{rsPathsRawBoostingDropFive}} | {{rsPathsRawBoostingDropTen}} |

**`fragility_across_paths`, log1p**

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | {{rsPathsLogRidgeRsquared}} | {{rsPathsLogRidgeTrainRsquared}} | {{rsPathsLogRidgeGap}} | {{rsPathsLogRidgeSpearman}} | {{rsPathsLogRidgeMae}} / {{rsPathsLogBaselineMae}} | {{rsPathsLogRidgeDropFive}} | {{rsPathsLogRidgeDropTen}} |
| Lasso | {{rsPathsLogLassoRsquared}} | {{rsPathsLogLassoTrainRsquared}} | {{rsPathsLogLassoGap}} | {{rsPathsLogLassoSpearman}} | {{rsPathsLogLassoMae}} / {{rsPathsLogBaselineMae}} | {{rsPathsLogLassoDropFive}} | {{rsPathsLogLassoDropTen}} |
| Elastic net | {{rsPathsLogElasticNetRsquared}} | {{rsPathsLogElasticNetTrainRsquared}} | {{rsPathsLogElasticNetGap}} | {{rsPathsLogElasticNetSpearman}} | {{rsPathsLogElasticNetMae}} / {{rsPathsLogBaselineMae}} | {{rsPathsLogElasticNetDropFive}} | {{rsPathsLogElasticNetDropTen}} |
| Random forest | {{rsPathsLogForestRsquared}} | {{rsPathsLogForestTrainRsquared}} | {{rsPathsLogForestGap}} | {{rsPathsLogForestSpearman}} | {{rsPathsLogForestMae}} / {{rsPathsLogBaselineMae}} | {{rsPathsLogForestDropFive}} | {{rsPathsLogForestDropTen}} |
| Gradient boosting | {{rsPathsLogBoostingRsquared}} | {{rsPathsLogBoostingTrainRsquared}} | {{rsPathsLogBoostingGap}} | {{rsPathsLogBoostingSpearman}} | {{rsPathsLogBoostingMae}} / {{rsPathsLogBaselineMae}} | {{rsPathsLogBoostingDropFive}} | {{rsPathsLogBoostingDropTen}} |

**`fragility_across_regimes`, raw** (the charter's definition)

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | {{rsRegimesRawRidgeRsquared}} | {{rsRegimesRawRidgeTrainRsquared}} | {{rsRegimesRawRidgeGap}} | {{rsRegimesRawRidgeSpearman}} | {{rsRegimesRawRidgeMae}} / {{rsRegimesRawBaselineMae}} | {{rsRegimesRawRidgeDropFive}} | {{rsRegimesRawRidgeDropTen}} |
| Lasso | {{rsRegimesRawLassoRsquared}} | {{rsRegimesRawLassoTrainRsquared}} | {{rsRegimesRawLassoGap}} | {{rsRegimesRawLassoSpearman}} | {{rsRegimesRawLassoMae}} / {{rsRegimesRawBaselineMae}} | {{rsRegimesRawLassoDropFive}} | {{rsRegimesRawLassoDropTen}} |
| Elastic net | {{rsRegimesRawElasticNetRsquared}} | {{rsRegimesRawElasticNetTrainRsquared}} | {{rsRegimesRawElasticNetGap}} | {{rsRegimesRawElasticNetSpearman}} | {{rsRegimesRawElasticNetMae}} / {{rsRegimesRawBaselineMae}} | {{rsRegimesRawElasticNetDropFive}} | {{rsRegimesRawElasticNetDropTen}} |
| Random forest | {{rsRegimesRawForestRsquared}} | {{rsRegimesRawForestTrainRsquared}} | {{rsRegimesRawForestGap}} | {{rsRegimesRawForestSpearman}} | {{rsRegimesRawForestMae}} / {{rsRegimesRawBaselineMae}} | {{rsRegimesRawForestDropFive}} | {{rsRegimesRawForestDropTen}} |
| Gradient boosting | {{rsRegimesRawBoostingRsquared}} | {{rsRegimesRawBoostingTrainRsquared}} | {{rsRegimesRawBoostingGap}} | {{rsRegimesRawBoostingSpearman}} | {{rsRegimesRawBoostingMae}} / {{rsRegimesRawBaselineMae}} | {{rsRegimesRawBoostingDropFive}} | {{rsRegimesRawBoostingDropTen}} |

**`fragility_across_regimes`, log1p — the only specification that holds together**

| Model | R² | train R² | gap | ρ | MAE model / base | R² drop-5 | R² drop-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | {{rsRegimesLogRidgeRsquared}} | {{rsRegimesLogRidgeTrainRsquared}} | {{rsRegimesLogRidgeGap}} | {{rsRegimesLogRidgeSpearman}} | {{rsRegimesLogRidgeMae}} / {{rsRegimesLogBaselineMae}} | {{rsRegimesLogRidgeDropFive}} | {{rsRegimesLogRidgeDropTen}} |
| Lasso | {{rsRegimesLogLassoRsquared}} | {{rsRegimesLogLassoTrainRsquared}} | {{rsRegimesLogLassoGap}} | {{rsRegimesLogLassoSpearman}} | {{rsRegimesLogLassoMae}} / {{rsRegimesLogBaselineMae}} | {{rsRegimesLogLassoDropFive}} | {{rsRegimesLogLassoDropTen}} |
| Elastic net | {{rsRegimesLogElasticNetRsquared}} | {{rsRegimesLogElasticNetTrainRsquared}} | {{rsRegimesLogElasticNetGap}} | {{rsRegimesLogElasticNetSpearman}} | {{rsRegimesLogElasticNetMae}} / {{rsRegimesLogBaselineMae}} | {{rsRegimesLogElasticNetDropFive}} | {{rsRegimesLogElasticNetDropTen}} |
| **Random forest** | **{{rsRegimesLogForestRsquared}}** | {{rsRegimesLogForestTrainRsquared}} | {{rsRegimesLogForestGap}} | {{rsRegimesLogForestSpearman}} | **{{rsRegimesLogForestMae}}** / {{rsRegimesLogBaselineMae}} | **{{rsRegimesLogForestDropFive}}** | {{rsRegimesLogForestDropTen}} |
| Gradient boosting | {{rsRegimesLogBoostingRsquared}} | {{rsRegimesLogBoostingTrainRsquared}} | {{rsRegimesLogBoostingGap}} | {{rsRegimesLogBoostingSpearman}} | {{rsRegimesLogBoostingMae}} / {{rsRegimesLogBaselineMae}} | {{rsRegimesLogBoostingDropFive}} | {{rsRegimesLogBoostingDropTen}} |

### 7.3 Bias or variance

Every model fits its own training folds better than it generalises, and the gap is *widest* for the
model with the most capacity. On the log-regimes target the conservative random forest has both the
best out-of-fold R² ({{rsRegimesLogForestRsquared}}) and by far the best trimming stability
({{rsRegimesLogForestDropFive}} at drop-5, against {{rsRegimesLogBoostingDropFive}} for gradient
boosting), while gradient boosting fits its training folds hardest
({{rsRegimesLogBoostingTrainRsquared}}, a gap of {{rsRegimesLogBoostingGap}}).

**Nothing underfits. The limitation is data variance, not model bias**, and no higher-capacity
learner was built.

### 7.4 Why the primary target is unstable

| Cause | Verdict | Evidence |
|---|---|---|
| Heavy-tailed target | **Confirmed, dominant** | Top 5 of {{rsPredictorRows}} rows own {{rsPathsRawTailShareFive}}% of the sum of squares (raw); skew {{rsPathsRawSkew}}, excess kurtosis {{rsPathsRawKurtosis}}. `log1p` reduces the top-5 share to {{rsPathsLogTailShareFive}}%. |
| Multicollinearity | **Present, secondary** | Condition number {{rsFeatureCondition}}; `{{rsFeatureWorstPairA}}` ↔ `{{rsFeatureWorstPairB}}` at \|r\| = {{rsFeatureWorstPairR}}. Destabilises the linear models; leaves the tree models alone. |
| Influential observations | **Confirmed** | Removing one row (`{{rsPathsRawInfluenceWorst}}`) moves out-of-fold R² by {{rsPathsRawInfluenceLargest}}. |
| Insufficient sample | **Confirmed contributor** | Learning curve still climbing at n = {{rsCurveSizeAtHundred}}: ρ {{rsPathsLogCurveAtForty}} → {{rsPathsLogCurveAtHundred}} on the log-path target. Has not saturated. |
| Genuinely weak relationship | **Partially rejected** | Permutation test, {{rsPermutations}} shuffles: p(ρ) = {{rsRegimesLogPermPSpearman}}, {{rsPathsLogPermPSpearman}}, {{rsRegimesRawPermPSpearman}} and {{rsPathsRawPermPSpearman}} across the four targets. But p(R²) reaches significance on only one — {{rsRegimesLogPermPRsquared}} for log-regimes, against {{rsPathsRawPermPRsquared}}, {{rsPathsLogPermPRsquared}} and {{rsRegimesRawPermPRsquared}} elsewhere. |

**The conclusion.** Strategy characteristics carry statistically detectable information about the
fragility *ranking* (ρ ≈ {{rsPathsRawPermSpearman}} to {{rsPathsLogPermSpearman}}, permutation
p = {{rsRegimesLogPermPSpearman}}–{{rsPathsRawPermPSpearman}}) and very weak information about its
*level*.

### 7.5 The full diagnostic table, all four targets

Published in full rather than for the two the argument turns on. A reader who suspects the
conclusion was reached by picking a favourable specification can check every one.

| Diagnostic | paths[raw] | paths[log1p] | regimes[raw] | regimes[log1p] |
|---|---:|---:|---:|---:|
| Skewness | {{rsPathsRawSkew}} | {{rsPathsLogSkew}} | {{rsRegimesRawSkew}} | {{rsRegimesLogSkew}} |
| Excess kurtosis | {{rsPathsRawKurtosis}} | {{rsPathsLogKurtosis}} | {{rsRegimesRawKurtosis}} | {{rsRegimesLogKurtosis}} |
| Top-5 rows' share of variance | {{rsPathsRawTailShareFive}}% | {{rsPathsLogTailShareFive}}% | {{rsRegimesRawTailShareFive}}% | {{rsRegimesLogTailShareFive}}% |
| Top-10 rows' share of variance | {{rsPathsRawTailShareTen}}% | {{rsPathsLogTailShareTen}}% | {{rsRegimesRawTailShareTen}}% | {{rsRegimesLogTailShareTen}}% |
| Best model by ρ | {{rsPathsRawBestModel}} | {{rsPathsLogBestModel}} | {{rsRegimesRawBestModel}} | {{rsRegimesLogBestModel}} |
| Best model by R² | {{rsPathsRawBestRsquaredModel}} | {{rsPathsLogBestRsquaredModel}} | {{rsRegimesRawBestRsquaredModel}} | {{rsRegimesLogBestRsquaredModel}} |
| Best R² | {{rsPathsRawBestRsquared}} | {{rsPathsLogBestRsquared}} | {{rsRegimesRawBestRsquared}} | {{rsRegimesLogBestRsquared}} |
| Observed ρ (permutation test) | {{rsPathsRawPermSpearman}} | {{rsPathsLogPermSpearman}} | {{rsRegimesRawPermSpearman}} | {{rsRegimesLogPermSpearman}} |
| Null \|ρ\| 95th percentile | {{rsPathsRawPermNullSpearman}} | {{rsPathsLogPermNullSpearman}} | {{rsRegimesRawPermNullSpearman}} | {{rsRegimesLogPermNullSpearman}} |
| p(ρ) | {{rsPathsRawPermPSpearman}} | {{rsPathsLogPermPSpearman}} | {{rsRegimesRawPermPSpearman}} | {{rsRegimesLogPermPSpearman}} |
| p(R²) | {{rsPathsRawPermPRsquared}} | {{rsPathsLogPermPRsquared}} | {{rsRegimesRawPermPRsquared}} | {{rsRegimesLogPermPRsquared}} |
| Largest single-row ΔR² | {{rsPathsRawInfluenceLargest}} | {{rsPathsLogInfluenceLargest}} | {{rsRegimesRawInfluenceLargest}} | {{rsRegimesLogInfluenceLargest}} |
| Most influential row | `{{rsPathsRawInfluenceWorst}}` | `{{rsPathsLogInfluenceWorst}}` | `{{rsRegimesRawInfluenceWorst}}` | `{{rsRegimesLogInfluenceWorst}}` |

**Note the pattern in the last two rows of the permutation block.** p(ρ) reaches significance on
all four specifications; p(R²) on one. The rank is learnable and the level is not, and that holds
whichever target definition and transform you choose.

Learning curves, ρ against training-set size (repeated subsampling, {{rsPredictorRows}} rows
available so the largest planned size was not reachable):

| n | paths[raw] | paths[log1p] | regimes[raw] | regimes[log1p] |
|---:|---:|---:|---:|---:|
| {{rsCurveSizeAtForty}} | {{rsPathsRawCurveAtForty}} | {{rsPathsLogCurveAtForty}} | {{rsRegimesRawCurveAtForty}} | {{rsRegimesLogCurveAtForty}} |
| {{rsCurveSizeAtSixty}} | {{rsPathsRawCurveAtSixty}} | {{rsPathsLogCurveAtSixty}} | {{rsRegimesRawCurveAtSixty}} | {{rsRegimesLogCurveAtSixty}} |
| {{rsCurveSizeAtEighty}} | {{rsPathsRawCurveAtEighty}} | {{rsPathsLogCurveAtEighty}} | {{rsRegimesRawCurveAtEighty}} | {{rsRegimesLogCurveAtEighty}} |
| {{rsCurveSizeAtHundred}} | {{rsPathsRawCurveAtHundred}} | {{rsPathsLogCurveAtHundred}} | {{rsRegimesRawCurveAtHundred}} | {{rsRegimesLogCurveAtHundred}} |

Three of the four curves rise monotonically to the largest size available. **The curve has not
saturated, and more strategies is the one intervention this diagnosis supports.** Generating them
means reopening the generator and re-running the audit and stress pipeline for the new candidates
— a separate undertaking, and left as future work.

### 7.6 Factor exposures and feature importances

Factor exposures come from **one joint regression on all factors at once**, not from separate
univariate fits. The joint design's condition number is {{rsFactorCondition}} across
{{rsFactorCount}} factors, with a maximum pairwise correlation of {{rsFactorWorstPair}} — well
inside the range where joint coefficients are stable. Univariate betas are retained in the feature
table as a labelled sensitivity and are not offered to the models.

Top permutation importances on the primary target, after deduplication:

| Rank | Feature | Increase in out-of-fold MAE when shuffled |
|---:|---|---:|
| 1 | `{{rsImportanceFirstName}}` | {{rsImportanceFirstValue}} |
| 2 | `{{rsImportanceSecondName}}` | {{rsImportanceSecondValue}} |
| 3 | `{{rsImportanceThirdName}}` | {{rsImportanceThirdValue}} |
| 4 | `{{rsImportanceFourthName}}` | {{rsImportanceFourthValue}} |
| 5 | `{{rsImportanceFifthName}}` | {{rsImportanceFifthValue}} |

This ranking **changed completely** once duplicates were removed and joint betas replaced
univariate ones. The earlier ranking was substantially an artefact of duplicated rows. See
`CORRECTIONS.md`.

---

## 8. Stress narratives

{{rsNarrativeCount}} narratives generated by `{{rsNarrativeModel}}` at {{rsNarrativeSeconds}}
seconds each, temperature 0, seed 42. The prompt contains only measured facts, and those facts are
stored beside each narrative so any claim can be audited against them.

**This is the weakest component of the benchmark and is presented as illustrative.** The narratives
are factually correct but near-vacuous: they restate the fragility measurement rather than explain
it, and none used the turnover, holding-period or concentration facts supplied to them. One
narrative describes a strategy as most reliable in a regime in which it loses money — fragility
measures consistency, not quality, and the narrative layer does not enforce that distinction.

---

## 9. Fragility distribution

Median across-regime fragility {{rsMedianFragilityRegimes}}, interquartile range
[{{rsIqrFragilityRegimesLow}}, {{rsIqrFragilityRegimesHigh}}], n = {{rsNPrimary}}.

Fragility is a ratio with a mean in its denominator, so strategies whose mean performance is near
zero carry a large fragility for an arithmetic reason. {{rsNNearZeroMean}} are flagged and reported
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

`scripts/run_stress_tier1.py` is the only expensive step ({{rsTierOneCpuHours}} CPU-hours). Every
other step completes in minutes.
