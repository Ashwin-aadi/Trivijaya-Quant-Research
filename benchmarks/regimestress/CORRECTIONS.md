# RegimeStress — corrections log

Every defect found in this benchmark, what it changed, and what it did not. Nothing here is
removed or softened. A benchmark whose corrections are invisible is a benchmark whose results
cannot be trusted, and the two entries below both moved a *published* number in an unflattering
direction.

| Version | Date | Nature |
|---|---|---|
| **1.0** | **2026-08-02** | First freeze. Carries all corrections below, found before the freeze and applied to it. |

> **On the provenance of the numbers in this file.** Post-correction figures are generated from the
> artifacts and appear in `RESULTS.md`; they are repeated here for readability. **Pre-correction
> figures are transcribed from the superseded checkpoint report and cannot be regenerated** — the
> configuration that produced them no longer exists in the pipeline. That asymmetry is unavoidable
> in a corrections log and is flagged rather than hidden. Where a before/after pair appears below,
> only the "after" column has machine-checkable provenance.

---

## C1 — Duplicate leakage: identical strategies on both sides of a cross-validation split

**Severity: material. It was worth 0.238 of R², and it was the difference between a positive
headline and a null result.**

### What was found

The generated corpus contains strategies whose *realised net return series are identical on every
session*, to within 1e-12. Different source code, different stated rationale, same behaviour: the
generator wrote the same strategy several times over.

| Quantity | Value |
|---|---:|
| Strategies compared | 150 |
| Clusters of identical series | 11 |
| Strategies inside a cluster | 27 |
| Largest cluster | 4 |
| Removed (alphabetically first retained per cluster) | 16 |

Cross-validation assumes rows are exchangeable and independent. They were not. A duplicated
strategy sat in a training fold while its twin was scored in the test fold, and the model was
rewarded for memorising a row it had already seen.

### What it cost

Same features, same folds, same seed, differing only in whether the duplicates are present:

| Configuration | n | Out-of-fold R² on the primary target |
|---|---:|---:|
| With duplicates | 125 | +0.262 |
| Without | 109 | +0.024 |
| **Leakage** | | **0.238** |

The Checkpoint 2.2 headline of R² = +0.410 **no longer exists.** That figure combined this leakage
with the feature defect in C2.

### How it was found

Not by a test. It was found by writing `scripts/deduplicate_corpus.py` in response to a PI
instruction to deduplicate before reporting feature importances — a methodological default the PI
set, not a bug report. The instruction was given for a different reason (importance stability) and
uncovered this.

### What was *not* done

**20 near-duplicate pairs (r ≥ 0.9999) were detected and deliberately left in.** Removing them
requires a correlation threshold, and any threshold chosen after seeing its effect on the result is
the exact pathology this lab was built to detect. They remain in the training set and are reported.

### Retained as a benchmark property

Duplicate structure is not noise to be cleaned away — it is a **fact about LLM strategy
generators**, and the benchmark records it. `duplicates.json` ships with the benchmark so a future
generator can be scored on how much of its output is redundant.

---

## C2 — A feature column that was another feature column

**Severity: material for the linear models, immaterial for the trees. Found by a diagnostic on its
first smoke run.**

`n_beta_sessions` is `n_sessions` under a different name. They correlate at exactly 1.000, and
together they drove the feature matrix's condition number to the order of 1e15 — a numerically
singular design presented to a ridge regression as though it were well posed.

Excluded from the feature set. `n_sessions` itself is **retained** despite being a confound
(strategies that go bankrupt have shorter series), because dropping it would hide the confound
rather than measure it.

After the fix the condition number is 253.2, with the worst remaining pair
`mean_herfindahl` ↔ `mean_largest_weight_share` at |r| = 0.995 — collinear, reported, and left
alone.

---

## C3 — The feature importances published at Checkpoint 2.2 were an artefact

Not a separate defect: a *consequence* of C1 and C2, recorded separately because the earlier
ranking was reported to the PI and would otherwise stand.

| Before (n = 125, univariate betas, duplicates present) | After (n = 109, joint betas, deduplicated) |
|---|---|
| `beta_relative_strength_vs_universe` +1.420 | `beta_long_term_reversal_756d` +0.408 |
| `beta_bollinger_reversion` +0.792 | `factor_r_squared` +0.280 |
| `beta_random_walk_baseline` +0.544 | `trading_session_rate` +0.187 |
| `n_entries` +0.356 | `mean_holding_period` +0.176 |
| `book_similarity_21d` +0.317 | `beta_inverse_volatility_weighted` +0.148 |

The old top feature is gone entirely, every magnitude fell by roughly a factor of three, and
structural characteristics — how often a strategy trades, how long it holds — replaced the beta
fingerprints.

---

## C4 — A methodological argument of ours that did not survive measurement

Not a code defect. Recorded because it is the kind of thing that usually goes unrecorded.

We argued for univariate factor regressions on the grounds that the factor proxies were too
collinear for a stable joint fit. **The PI's default was a joint regression, and the PI was right.**
The joint design's condition number is 12.0 across 10 factors, with a maximum pairwise correlation
of 0.946 — comfortably inside the range where joint coefficients are stable. The stated reason for
the univariate specification was not supported when it was finally measured.

Joint betas are now the primary specification. Univariate betas are retained in the feature table
under a `uni_` prefix as a labelled sensitivity and are not offered to the models.

---

## C5 — A diagnostic of ours that was wrong, caught by its own test

The leave-one-out influence diagnostic re-split the cross-validation folds after each row removal.
That moved every *surviving* row into a different fold, so the measured "influence" of removing a
row was dominated by reshuffling noise rather than by the row.

Caught by `test_a_single_extreme_row_is_identified_as_the_influential_one`, which plants one
extreme observation and asserts that it tops the ranking. It did not — an ordinary row did. Fixed
by materialising the fold assignment once and holding it fixed across all refits.

**The influence numbers reported anywhere before this fix are void.** The ones in `RESULTS.md` are
post-fix.

---

## What none of these corrections changed

The regime labels, the resampler, the moment validation, the Tier 1 experiment, the tier
comparison, and the shortcut validation are all untouched by C1–C5. Every correction lands inside
the fragility *predictor*, which is the benchmark's secondary contribution. The stress-testing
infrastructure — the part intended to last — was not affected.

That is stated as a fact, not as reassurance: a reader who distrusts the predictor results after
reading this file is reading it correctly, and should note that the predictor's final reported
conclusion is a null.
