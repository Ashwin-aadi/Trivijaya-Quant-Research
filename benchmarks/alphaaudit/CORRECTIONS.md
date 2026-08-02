# AlphaAudit — corrections log

Every correction applied to this benchmark after its first freeze, with what changed, what did
not, and why. A benchmark that is quietly edited is not a benchmark, so nothing here is removed:
superseded artifacts are retained beside their replacements under `runs/pooled/` with a
`_v1.0_superseded` suffix.

| Version | Date | Nature |
|---|---|---|
| 1.0 | 2026-08-01 | First freeze. |
| 1.1 | 2026-08-01 | Implementation bug in the abstention ranking. Semantic-inclusive configurations rerun. |
| **1.2** | **2026-08-02** | **Disclosure only, nothing rerun. 27 of the 174 survivors are not deterministic functions of their inputs.** |
| **1.3** | **2026-08-02** | **Disclosure only, nothing rerun. The corpus contains 11 clusters of behaviourally identical strategies, and one candidate is the equal-weight baseline.** |

---

## v1.2 — 27 survivors are not deterministic functions of their inputs

**Nothing was rerun and no number in this benchmark changed.** This entry exists because a
property of the corpus was discovered after the freeze which a reader is entitled to know, and
which the benchmark's own reproducibility claim does not survive without it.

### What was found

Run one of these strategies twice, on the same panel, with the same code, and it returns a
different Sharpe ratio. Measured across the whole 185-strategy population — the 174 survivors plus
the 11 standard factors — by `scripts/calibrate_tier1.py`, which writes
`data/processed/tier1_calibration.json`. That artifact is **not** committed: `/data/` is gitignored
repository-wide, so the file is regenerable rather than tracked, and the command that regenerates it
is given at the end of this section.

| Population | Deterministic | Not |
|---|---|---|
| Survivors | **147 of 174** | 27 |
| Standard factors | **11 of 11** | 0 |
| Total | **158 of 185** | 27 |

Every affected strategy is generator output. The hand-written fixtures are all clean, which is
itself informative: this is a property of what the model wrote, not of the engine or the harness.

### The two mechanisms, both confirmed in source

**Unseeded randomness — 26 of the 27.** These differ between two runs that are identical in every
respect including the hash seed. `candidate_1011` returned Sharpe −3.4386, −4.2835 and −1.6182 on
three consecutive runs *inside a single process*.

**Hash-order dependence — 27 of the 27**, overlapping heavily with the above, since anything with
an unseeded RNG also moves when re-run. The clearest instance is `candidate_002` line 42:

```python
volatility_factors = list(set(volatility_factors))[:10]
```

It selects an arbitrary ten of the qualifying symbols. Python randomises `set` iteration order per
process, so the strategy holds a different portfolio on every run. Its Sharpe was observed at
0.7652, 0.4566 and 0.6825 under `PYTHONHASHSEED` 0, 1 and 2.

The largest swing observed between two seeds is `candidate_1011` at **|Δ| = 2.5352**, followed by
`candidate_219` at 1.2663.

### Why the harness did not prevent it

`PYTHONHASHSEED` is set **nowhere in this repository**, so the corpus backtests ran with hash
randomisation active and a different seed in every worker process. `src/common/seeding.py` seeds
Python's `random` and NumPy, but neither reaches a strategy that constructs its own RNG, and
neither can affect hash order — that must be fixed before the interpreter starts. Charter RULE 6
requires that no stochastic operation be unseeded; for these 27 strategies that requirement was
not met, and the failure was in the harness's coverage rather than in any single script.

### What this does and does not invalidate

**Affected.** The recorded net Sharpe of each of the 27 is **one draw from a distribution, not a
fixed quantity**. Because the abstention curves are computed from those Sharpes, the AUAP figures
in `RESULTS.md` §9 carry that noise too, and their trailing digits should not be read as exact.

**Not affected.**

* **Survivor membership.** A survivor is a candidate no auditor layer rejected, and the auditors
  read source code, not performance. The single performance touch in `scripts/tag_survivors.py` is
  a flatness gate at `FLAT_TOLERANCE = 1e-9`; the smallest absolute Sharpe among the 27 is
  **0.0111**, seven orders of magnitude clear of it. No membership decision is close.
* **The headline conclusion.** The finding is that no auditor configuration beats random rejection.
  Additional noise in the performance figures makes an ordering *harder* to detect, not easier; it
  cannot manufacture the result "the auditor works". The null stands.
* Leak-class rates, the funnel, Cohen's κ, the deflation figures and the cost model. None reads a
  strategy's realised performance.

### Why nothing was rerun

The holdout is permanently closed (§10 of `RESULTS.md`) and the PI declined to reopen it. A rerun
is therefore unavailable, and it is also unwarranted: the conclusion does not turn on the affected
quantities, and re-running under a pinned seed would replace one arbitrary draw with another
arbitrary draw rather than with a correct value. **The defect is in the strategies, not in the
measurement of them.** Pinning the seed would make the benchmark reproducible without making those
27 strategies well-defined.

### The consequence for anything downstream

Fragility, the quantity Project 2 measures, is a variance across counterfactual histories. A
strategy whose score already varies for no reason contributes variance that is arithmetically
indistinguishable from the regime response being measured. **These 27 are therefore excluded from
Project 2's population** (PI decision, 2026-08-02), which is recorded here rather than only in P2
because it changes what the survivor set means downstream.

That exclusion is **not performance-neutral, and must never be reported as though it were**:

| Group | n | Mean net Sharpe |
|---|---|---|
| Excluded (nondeterministic) | 27 | **−0.6722** |
| Retained (deterministic) | 147 | **−0.1320** |
| All survivors | 174 | −0.2158 |

Dropping them raises the population's mean Sharpe by 0.08 and its median to +0.4612. The exclusion
rule is mechanistic — a strategy must be a deterministic function of its inputs — and was not
chosen on performance, but it correlates with performance, plausibly because arbitrary symbol
selection produces near-random portfolios with high turnover and therefore heavy cost drag. Any
downstream figure computed on the 147 must be accompanied by this table.

### How to check this yourself

```bash
python scripts/calibrate_tier1.py --workers 24     # ~17 minutes, 740 backtests
```

Group 0 is the decisive one: the same panel, the same hash seed, two independent runs. Anything
that moves there is nondeterministic and no argument about floating point can explain it.

---

## v1.1 — sign error in the auditor-confidence ranking

### What was wrong

`soundness()` in `scripts/run_ablation.py` converted each layer's stored `confidence` into a
ranking score by negating it:

```python
return -float(record.get("confidence", 0.0))     # v1.0
```

This assumed every layer stores *confidence that something is wrong*. Two of the three do:

* **static** stores a finding count — `0.0` when clean, `1.0`/`2.0`/`3.0` when flagged;
* **statistical** stores a constant `1.0`, on records it rejects without exception (631 of 631).

The **semantic** layer does not. It stores the model's confidence **in its own label**: about
`0.95` for a confident `consistent` and about `0.85` for a confident defect. Negating that ranked
the strategies the semantic layer had *rejected* above the ones it had *cleared* — the ranking was
inverted, so at low coverage the pipeline retained precisely the candidates the layer had
condemned.

Measured on the 225 rankable candidates: Spearman correlation between the v1.0 and v1.1 semantic
ranking is **−0.972**, and the retained set at 5% coverage went from **11 of 11 semantically
rejected** to **0 of 11**.

### Why no test caught it

Each layer's own tests check its labels, which were correct. `tests/eval/test_abstention.py`
checks that the curve is computed correctly from a ranking, which it was. The defect lived only in
the join between them — a ranking computed correctly from a quantity that meant the opposite of
what the calling code believed. `tests/eval/test_soundness_sign.py` now asserts that join
directly, per layer, in the direction a reader can check by eye.

### The fix

Sign comes from the layer's own `rejected` verdict rather than from an assumption about what
`confidence` means:

```python
confidence = float(record.get("confidence", 0.0))
return -confidence if record.get("rejected", True) else confidence     # v1.1
```

Unscored candidates still contribute `0.0`, which now sits between a confident acceptance and a
confident rejection.

### What was rerun, and what was not

The correction is a general rule, but on this corpus it is **provably confined to the semantic
layer**, because the static layer records `rejected` exactly when its finding count is non-zero and
the statistical layer rejects every record it scores. Both therefore produce values identical to
v1.0 under the new rule. This is a property of the frozen data, not an argument, and
`tests/eval/test_soundness_sign.py::test_the_correction_leaves_the_other_layers_bit_identical`
checks it against the data.

The three non-semantic configurations were recomputed anyway and came back **bit-identical**
(∆AUAP = +0.0000 in every case), which is a stronger demonstration than skipping them.

Nothing else was touched: no methodology, thresholds, prompts, model outputs, backtests, universe,
cost model, trial counter, or holdout contents. The corpus, the audit labels and the two
performance files are byte-for-byte the ones v1.0 used. The only difference between the runs is the
two lines above.

### Holdout AUAP — v1.0 vs v1.1

Population n = 225, fixed by development eligibility. Random-rejection 95% interval
**[−1.2208, −0.8600]**, unchanged (it does not depend on the ranking).

| Layers | AUAP v1.0 | **AUAP v1.1** | ∆ | P(0.05) v1.0 | **P(0.05) v1.1** | Position vs random |
|---|---|---|---|---|---|---|
| semantic | −1.1441 | **−1.1967** | −0.0526 | −0.5147 | **−1.1347** | inside |
| static | −1.2529 | **−1.2529** | +0.0000 | −1.1176 | **−1.1176** | below |
| static + semantic | −1.2249 | **−1.2782** | −0.0533 | −0.8660 | **−1.1805** | below |
| statistical | −1.3497 | **−1.3497** | +0.0000 | −3.0643 | **−3.0643** | below |
| semantic + statistical | −1.2344 | **−1.3777** | −0.1433 | −0.5147 | **−3.2931** | below |
| static + semantic + statistical | −1.2849 | **−1.4184** | −0.1336 | −0.8660 | **−3.2931** | below |
| static + statistical | −1.4339 | **−1.4339** | +0.0000 | −3.2931 | **−3.2931** | below |

### Development AUAP — v1.0 vs v1.1 (diagnostic only)

Random band **[−0.3706, −0.0270]**.

| Layers | AUAP v1.0 | **AUAP v1.1** | ∆ | P(0.05) v1.0 | **P(0.05) v1.1** |
|---|---|---|---|---|---|
| semantic | −0.2712 | **−0.2917** | −0.0205 | −0.1088 | **+0.2910** |
| static + semantic | −0.3055 | **−0.3310** | −0.0255 | −0.0918 | **+0.3142** |
| static | −0.3327 | **−0.3327** | +0.0000 | +0.2073 | **+0.2073** |
| statistical | −0.3995 | **−0.3995** | +0.0000 | −1.3599 | **−1.3599** |
| semantic + statistical | −0.3290 | **−0.4124** | −0.0834 | −0.1088 | **−1.3494** |
| static + semantic + statistical | −0.3514 | **−0.4374** | −0.0860 | −0.0918 | **−1.3494** |
| static + statistical | −0.4550 | **−0.4550** | +0.0000 | −1.3494 | **−1.3494** |

### Effect on the conclusions

**The headline conclusion is unchanged: the null is not rejected.** No configuration beat random
rejection under v1.0 and none does under v1.1. The correction moved every affected number in the
direction that *strengthens* that finding, not away from it:

* The best holdout AUAP is now **−1.1967 (`semantic` alone)**, previously −1.1441, and it is still
  inside the random interval. The identity of the best configuration did not change.
* Most curves still **fall** as coverage tightens; with the ranking corrected they fall harder.

### A second error, found while checking the first

v1.0's prose stated that "five configurations sit inside the random interval; two sit below it",
and its table annotated only `static` and `static + statistical` as below. **That was a
miscount, present in v1.0 and unrelated to the ranking bug.** Six of the seven v1.0 AUAPs were
already below the interval's lower bound of −1.2208; the annotation was internally inconsistent,
marking `static` at −1.2529 as below while leaving `statistical` at −1.3497 unmarked. The count is
therefore **six of seven below the interval in both versions**, and only `semantic` alone inside
in both.

The correction to the ranking did not change that count. It is stated separately here so that
nothing in the v1.1 write-up claims credit for fixing an error the v1.1 ranking did not cause.
The v1.0 numbers themselves were correct; only the sentence counting them was wrong.

### Two consequences of the ranking correction, stated plainly

Neither is left for a reader to notice.

**The corrected ranking performs worse out of sample than the inverted one.** Every affected AUAP
moved down. It would be an overclaim to read that as the semantic layer being anti-informative:
both orderings sit inside or below a random interval that is 0.36 wide, and a difference of 0.05
between two null results is not a finding. The honest statement is that the layer carries no usable
out-of-sample ordering information in either direction.

**In sample, the correction does what a working auditor should do, and it does not survive.**
Development `P(0.05)` for `semantic` rose from −0.1088 to **+0.2910**, and for `static + semantic`
from −0.0918 to **+0.3142** — a correctly-oriented semantic layer visibly concentrates the better
strategies at tight coverage on the data it was developed against. The same ranking on the holdout
gives −1.1347 and −1.1805. That gap between an in-sample selectivity that looks real and an
out-of-sample one that does not is the paper's subject, arrived at here by accident.

### One further observation, not corrected

Under both v1.0 and v1.1 the **statistical** layer contributes no ordering information among the
candidates it scores: it rejects all 631 at confidence 1.0, so every scored candidate ties at
−1.0 and the layer's only discrimination is between scored and unscored. That is a genuine
weakness of the layer as implemented, but it is not an inversion and it was not in the scope
authorised for this correction. It is recorded here and left for the PI to rule on.

### Reproduction

```bash
python scripts/run_ablation.py --corpus runs/pooled/candidates
python scripts/run_ablation.py --corpus runs/pooled/candidates --holdout --authorised-by "<who, when>"
python scripts/plot_abstention.py --ablation runs/pooled/ablation_holdout.json
pytest tests/eval/test_soundness_sign.py
```

The holdout authorisation recorded in `runs/pooled/ablation_holdout.json` is the PI's of
2026-08-01. Under Rule 7 the holdout is otherwise closed; this evaluation was authorised
explicitly as the repair of an erroneous evaluation pipeline, and no tuning of any kind followed
it.

---

## v1.3 — the generated corpus is substantially redundant, and one candidate is the baseline

**Disclosure only. Nothing was rerun and no number in this benchmark changed.** This entry records
a property of the AlphaAudit *corpus* that was discovered while building RegimeStress, where it
had a measurable consequence. It says something about the generator, so it belongs here rather
than in the downstream benchmark that happened to find it.

### What was found

Of 150 comparable strategies drawn from this corpus, **11 clusters contain strategies whose
realised net return series are identical on every session**, to within 1e-12. Different source
code, different stated rationale, identical behaviour.

| Quantity | Value |
|---|---|
| Strategies compared | 150 |
| Clusters of identical series | 11 |
| Strategies inside a cluster | 27 |
| Largest cluster | 4 (`candidate_084`, `candidate_1266`, `candidate_204`, `candidate_996`) |
| Near-duplicate pairs (r >= 0.9999) not merged | 20 |

The generator does not merely produce weak strategies. **It produces the same weak strategies
repeatedly, under different rationales**, and the audit layers do not detect this because no layer
compares candidates against each other. A rationale–implementation check reads one strategy at a
time; a corpus of near-clones passes it individually.

### One cluster is the positive control

```
['candidate_344', 'equal_weight_universe']
```

**An AI-generated candidate is identical, in realised returns on every session, to the equal-weight
baseline it was supposed to beat.** The model rediscovered equal-weighting and presented it as a
novel strategy with its own economic rationale. `A_sem`, the semantic layer, is specified to flag
"unacknowledged known anomaly" — rediscovering a standard result and presenting it as new. It did
not flag this one.

That is a concrete, checkable miss by the semantic auditor, and it is recorded as such. It is one
case, found incidentally rather than by systematic search, so it is not a measured false-negative
rate and must not be read as one.

### Why it matters to this benchmark's headline null

The AlphaAudit paper reports that no auditor configuration beats random rejection at matched
coverage, and states that the null "remains confounded with corpus degeneracy and cannot separate
*'the auditor is uninformative'* from *'this corpus is too weak to test it'*."

**This finding is direct quantitative evidence for that confound.** A corpus in which 27 of 150
strategies are exact behavioural duplicates has fewer effective candidates than its count suggests,
and an abstention curve computed over it is ranking fewer distinct objects than the coverage axis
implies. The null result stands as reported; the confound is now measured rather than merely
suspected.

### Consequences downstream, disclosed

In RegimeStress the duplicates had to be removed before cross-validation, because a strategy in a
training fold with its twin in a test fold inflated out-of-fold R-squared by 0.238. The removal rule
retains the alphabetically first name in each cluster, chosen in advance so that the choice could
not be made on the outcome. **A consequence of that rule is that `equal_weight_universe` was
removed and `candidate_344` retained**, so any analysis splitting that population by name prefix
will count the equal-weight baseline as an AI-generated candidate. Stated here so it cannot
mislead.

### Reproduction

```bash
python scripts/deduplicate_corpus.py     # writes benchmarks/regimestress/duplicates.json
```

The clusters, the removals and the 20 detected-but-retained near-duplicate pairs are all in that
artifact, which is committed.
