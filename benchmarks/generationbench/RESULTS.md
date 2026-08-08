# GenerationBench — results

Every number below is computed by `scripts/meta_analysis.py` from committed artefacts and written to
`META.json`. None is transcribed by hand. Regenerate with:

```
python scripts/meta_analysis.py
```

**Scope.** 2,495 strategies from six generation paradigms on one local 7B model, plus 60 strategies
from three frontier models, all pushed through the same frozen evaluation stack — the AlphaAudit
auditor, the RegimeStress suite, the FlowState capacity model. No stage's parameters were touched.

**The holdout was opened once per arm on 2026-08-08**, under PI authorisation, after RULE 7's three
conditions were verified in writing. It is now closed. No number here was tuned after that read.

---

## The eight findings

### 1. Sophistication does not survive compute matching

**Zero of five scaffolded paradigms beat plain prompting on yield when plain prompting is given the
same token budget and allowed to keep its best of k.**

| paradigm | k | plain prompting at the same budget | the paradigm | verdict |
|---|---|---|---|---|
| chain-of-thought | 2 | **27.1%** (n=775 blocks) | 11.5% (n=800) | loses |
| planning | 3 | **39.1%** (n=516) | 6.8% (n=365) | loses |
| reflection | 3 | **39.1%** (n=516) | 10.1% (n=288) | loses |
| graph-of-thoughts | 6 | **64.3%** (n=258) | 8.6% (n=152) | loses |
| Monte Carlo tree search | 14 | **89.1%** (n=110) | 58.3% (n=60) | loses |

The claim is deliberately narrow, and the narrower version is the defensible one:

> **Under compute-matched evaluation, no tested scaffolded paradigm improved yield over plain
> prompting.**

**What this does not say:** that scaffolding makes things worse. The experiment does not establish
that. It establishes that the extra compute would have been better spent on more simple attempts.

*Why the control is what it is:* a five-agent committee that beats one prompt has not been shown to
be better, because it thought five times as hard. The honest control is the simple method run five
times, keeping its best. Half the published work in this area omits it.

### 2. Frontier models fix the code, not the alpha

**Every frontier model wrote code that ran. None of them wrote code that made money.**

**The front of the funnel — transformed:**

| generator | n | executes | takes a position | fails static leakage screen |
|---|---|---|---|---|
| local 7B, plain prompting | 830 | 40.7% | **14.6%** | 13.5% |
| GPT | 20 | **100%** | **100%** | 0.0% |
| Claude | 20 | **100%** | **100%** | 0.0% |
| Gemini | 20 | **100%** | **100%** | 5.0% |

A 6.8× improvement in position-taking, and leakage by naive coding all but disappears.

**The back of the funnel — unchanged, or slightly worse:**

| generator | holdout median Sharpe | fraction with positive holdout Sharpe | AUAP combinations beating random |
|---|---|---|---|
| local 7B | −0.616 (n=127) | 27.6% | 0 of 7 |
| GPT | −1.149 (n=20) | 40.0% | 0 of 7 |
| Claude | −0.770 (n=20) | 35.0% | 0 of 7 |
| Gemini | −1.149 (n=20) | 30.0% | 0 of 7 |

**Read both columns.** Median holdout Sharpe is *worse* for all three frontier arms; the fraction of
strategies with a positive holdout Sharpe is *better* for all three. These are not contradictory —
the local figure is computed over 127 strategies that had already been filtered by having traded at
all, while the frontier figure covers all 20. **The honest summary is that frontier capability did
not produce out-of-sample profitability, not that it made it worse.**

> **Frontier capability moves the engineering, not the economics.** The hard part of quantitative
> research was never writing code that runs.

**Not compute-matched.** 20 frontier draws against 830 local ones at whatever each cost. Every figure
in this section is a *rate* claim and none is a RULE 11 win.

### 3. The auditor's ranking never beat a coin flip — 70 times out of 70

**Across ten independent corpora and seventy layer combinations, not one beat random rejection at
matched coverage on the holdout.**

| corpus | n ranked | combinations beating random | best AUAP |
|---|---|---|---|
| AlphaAudit pooled corpus | 225 | 0 of 7 | −1.1967 |
| paradigm G1 plain | 121 | 0 of 7 | −1.3754 |
| paradigm G2 chain-of-thought | 92 | 0 of 7 | −1.6264 |
| paradigm G4 planning | 25 | 0 of 7 | −0.6056 |
| paradigm G5 reflection | 29 | 0 of 7 | −1.4013 |
| paradigm G6 graph-of-thoughts | 13 | 0 of 7 | −1.0880 |
| paradigm G7 MCTS | 35 | 0 of 7 | +0.0124 |
| generator GPT | 20 | 0 of 7 | −0.9717 |
| generator Claude | 20 | 0 of 7 | −0.6083 |
| generator Gemini | 20 | 0 of 7 | −1.7411 |
| **total** | | **0 of 70** | |

**Why this is the programme's strongest result rather than its most embarrassing.** AlphaAudit
published this null on one corpus, where the obvious objection was *"you measured your own small
model."* It now survives four generators and six generation methodologies. A metric that beat random
on some corpora and not others would have been evidence about the corpora. Failing everywhere,
identically, is evidence about the metric.

**What it does not say.** The auditor is not useless. Its static layer catches leakage, measured
separately and successfully. What fails is the claim that its *confidence ordering* predicts
out-of-sample performance. Refusing to act on what it doubts does not leave you with something
better.

### 4. The statistical layer is saturated, not strict

**It rejects 100% of every corpus, at a recorded confidence of exactly 1.0.**

| arm | scored | rejected |
|---|---|---|
| G1 | 338 | 338 |
| G2 | 294 | 294 |
| G4 | 83 | 83 |
| G5 | 98 | 98 |
| G6 | 45 | 45 |
| G7 | 59 | 59 |

Pooled trial count **3,499**. The single highest Deflated Sharpe probability anywhere in the corpus is
**0.17**, in G7.

**We cannot tell whether this is correct multiple-testing correction or a test that has run out of
resolution.** AlphaAudit reached the same wall at N=1,887 and said the same thing. The deflation term
grows with the trial count while the sample length stays fixed, so at some ratio the test rejects
everything regardless of merit — and no independent calibration exists showing the implementation
behaves correctly under *this* trial-generation process.

**A layer that rejects everything carries no ordering information.** Every candidate ties. It is
therefore reported beside the funnel rather than inside it, which is a definitional choice made
because it produces a non-empty answer, and is flagged as one. AlphaAudit made and flagged the same
choice before its own holdout evaluations, so it is inherited, not invented after seeing P4's results.

### 5. MCTS's advantage has a mechanism, and the mechanism is that it cheats at the front

**Corrected from the Checkpoint 4.2 report, which said no mechanism existed.** One does, and it is
documented in the code.

G7 executes at **98.3%** and takes a position at **58.3%**, against 22.7–40.7% and 6.8–14.6% for
every other arm. Its full-stack survival is **43.33%** against 5.26–8.55%.

`src/generate/paradigms/fitness.py` runs **a backtest, net of Indian transaction costs, during the
search**. G7 is the only arm that evaluates its own intermediate output before emitting it. It
therefore does not emit strategies that fail to run — it has already discarded them.

This is neither an artefact nor a paradigm effect. It is **a different information budget**, and the
pre-registration flagged the confound before any data existed:

> *"this arm receives cost information no other arm receives, so part of any advantage it shows may
> be cost-optimisation rather than search. That confound is pre-registered."*

The honest trial counter charges it accordingly: **736 trials for 60 emitted draws.** And it still
loses its compute-matched comparison, 58.3% against 89.1%.

The frozen stack is not reachable from the fitness function — no auditor, no fragility model, no
capacity model — and the holdout is structurally unreachable, with no code path that could load it.

### 6. A redundancy of zero can mean nothing at all

**Graph-of-thoughts scored 0.0% redundancy. An arm that size shows zero duplicates 88% of the time
even if it repeats itself exactly as much as plain prompting does.** *(Exploratory, not
pre-registered.)*

Reference duplicate rate from G1 and G2 pooled: 19 duplicate pairs in 9,795 pairs, or 1 in 516.

| arm | traded | pairs | R(G) | P(zero duplicates at the reference rate) | informative? |
|---|---|---|---|---|---|
| G1 | 115 | 6,555 | 13.0% | 0.0% | yes |
| G2 | 81 | 3,240 | 17.3% | 0.2% | yes |
| G4 | 23 | 253 | 0.0% | **61.2%** | no |
| G5 | 25 | 300 | 8.0% | 55.9% | — |
| G6 | 12 | 66 | 0.0% | **88.0%** | no |
| G7 | 35 | 595 | 14.3% | 31.5% | — |

Duplicates are counted over *pairs*, which grow quadratically, so the equal-token design gave the
expensive arms roughly one per cent of the cheap arms' opportunity to show a duplicate at all.
**Neither zero in this table is evidence of diversity.**

The figure is conservative in the unhelpful direction: pairs are treated as independent though
duplicates arrive in clusters, which if corrected would push P(zero) higher still.

### 7. One traded strategy in seven is another paradigm's strategy

**43 of 291 pooled traded strategies (14.8%) sit in a duplicate cluster spanning more than one arm.
13 of 20 pooled clusters span arms; the widest spans four.** *(Exploratory, not pre-registered.)*

| arm | duplicated in another arm |
|---|---|
| G5 reflection | **24.0%** (n=25) |
| G2 chain-of-thought | 19.8% (n=81) |
| G1 plain | 13.9% (n=115) |
| G4 planning | 13.0% (n=23) |
| G7 MCTS | 5.7% (n=35) |
| G6 graph-of-thoughts | 0.0% (n=12) |

R(G) as pre-registered measures whether a paradigm repeats *itself*. It cannot see this. Reflection's
rewrites converge on output plain prompting reached anyway — which, with reflection also being the
worst arm out of sample and the worst against its control, is three independent measurements pointing
the same way.

**A correction worth recording.** The stratified capacity table showed identical capacities across
arms — 3.85 crore in five of six, 0.19 crore in four — covering 139 strategies. That looked like mass
cross-arm duplication and was not reported. Tested against the actual definition of a duplicate, the
number is 43. Capacity is quantised enough that unrelated strategies land on the same figure.

### 8. Every enormous capacity was an empty account

**The three largest deployment capacities in the corpus belong to strategies that hold almost no
book.**

| strategy | binding capacity | gross exposure | names held |
|---|---|---|---|
| G6 candidate_144 | ₹1,283 crore | **0.0150** | 60 |
| G2 candidate_435 | ₹727 crore | **0.0028** | 100 |
| G7 candidate_012 | ₹372 crore | **0.0300** | 20 |
| G1 candidate_240 *(normal)* | ₹23.65 crore | 1.0000 | 5 |

These are not the floating-point residue defect FlowState repaired at Checkpoint 3.3 — that guard is
1e-9 and these trade four orders of magnitude above it. They are genuinely trading, and leaving
97–99.7% of the account in cash.

**Capacity is defined per rupee of AUM, not per rupee deployed.** A strategy that invests nothing
reports an enormous capacity that is arithmetically correct and substantively empty.

Stratified (near-cash = median gross exposure below 0.10, an analyst-defined reporting boundary, never
an exclusion):

| arm | deployed n | median crore | max crore | near-cash n | near-cash median crore |
|---|---|---|---|---|---|
| G1 | 115 | 2.60 | 23.65 | 6 | 1.16 |
| G2 | 84 | 1.77 | 61.35 | 8 | 1.62 |
| G4 | 25 | 3.85 | 36.88 | 0 | — |
| G5 | 25 | 1.38 | 9.86 | 4 | 1.04 |
| G6 | 11 | 9.91 | 36.88 | 2 | 641.98 |
| G7 | 33 | 5.16 | 36.88 | 2 | 282.17 |

**The confound was confined to the maxima.** G2's max falls 726.86 → 61.35, G6's 1283.04 → 36.88,
G7's 371.87 → 36.88. Medians move almost not at all, so the cross-arm ordering stands as measured.

**FlowState's five validation factors were all fully invested, so the validation set could not reach
this path.** The machine-written corpus reached it immediately. This is the second time the corpus has
exposed a gap the validation set could not, which is evidence for the standing
build → validate → apply → freeze → write process rather than against it.

Every capacity figure here is a **constraint** figure, never an impact-erosion figure. FlowState
measured that daily bars cannot identify a transient impact function.

---

## The full table

| | G1 plain | G2 CoT | G4 planning | G5 reflection | G6 GoT | G7 MCTS |
|---|---|---|---|---|---|---|
| draws | 830 | 800 | 365 | 288 | 152 | 60 |
| output tokens | 407,588 | 408,215 | 409,161 | 408,324 | 410,177 | 408,053 |
| tokens per draw | 491 | 510 | 1,121 | 1,418 | 2,699 | 6,801 |
| executes | 40.7% | 36.8% | 22.7% | 34.0% | 29.6% | 98.3% |
| takes a position | 14.6% | 11.5% | 6.8% | 10.1% | 8.6% | 58.3% |
| fails static screen | 13.5% | 12.8% | 18.1% | 16.3% | 16.4% | 8.3% |
| fails semantic screen | 10.0% | 10.5% | 13.2% | 13.5% | 8.6% | 8.3% |
| full-stack survival | 8.55% | 8.00% | 5.48% | 6.25% | 5.26% | 43.33% |
| development median Sharpe | 0.366 | 0.449 | 0.768 | −0.589 | 0.108 | 0.750 |
| **holdout median Sharpe** | **−0.616** | **−0.051** | **−0.060** | **−1.151** | **−0.507** | **+0.047** |
| holdout fraction positive | 27.6% | 47.9% | 48.3% | 6.5% | 28.6% | 60.0% |
| trials charged | 844 | 811 | 370 | 582 | 156 | 736 |
| PBO | 0.114 | 0.162 | 0.139 | 0.090 | 0.104 | 0.179 |
| knife-edge rate | 17.4% | 3.7% | 4.3% | 32.0% | 8.3% | 8.6% |
| fragility, tier 2, median | 0.539 | 0.615 | 0.523 | 1.035 | 0.430 | 0.615 |
| capacity, deployed median | ₹2.60 cr | ₹1.77 cr | ₹3.85 cr | ₹1.38 cr | ₹9.91 cr | ₹5.16 cr |
| R(G) redundancy | 13.0% | 17.3% | 0.0% | 8.0% | 0.0% | 14.3% |
| duplicated in another arm | 13.9% | 19.8% | 13.0% | 24.0% | 0.0% | 5.7% |
| AUAP beats random | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 | 0/7 |

Matched to **0.6%** on generated tokens — a spread of 2,589 on a base of ~408,000. That is the figure
RULE 11 requires and the one the primary comparison rests on.

---

## What is not comparable to what

1. **Fragility across projects.** The frontier arms carry tier 1 across-paths fragility at 100
   synthetic price panels; P4 carries tier 2 across-regimes at 1,000 resampled return paths.
   RegimeStress measured tier agreement on fragility at Spearman **0.620** and published that tier 2
   cannot substitute for tier 1 when fragility is the quantity of interest. Tier 1 was not affordable
   here — RegimeStress spent 124.5 CPU-hours on 125 strategies, so P4's 315 would cost upwards of 250.
   **The two are never placed in one table.** The justification is affordability, not agreement.
2. **The generator axis is not compute-matched.** 20 frontier draws against 830 local. Rate claims only.
3. **P4 capacity uses the unfiltered price panel; frontier capacity uses the truncated one**
   (FlowState `CORRECTIONS.md` C7). Capacity appeared insensitive to this on the one arm tested; it is
   not asserted for the others.
4. **Sample sizes differ by up to 40×.** G6 ranks 13 candidates for AUAP; G4 ranks 25. Those rows are
   consistent with a null and would be consistent with a great deal else.

## Gaps in what was tested

- **No retrieval-augmented arm.** Dropped before any data existed: indexing a factor-literature
  corpus is a project, and the data budget is ₹0. A hole in the paradigm space, reported as one.
- **No multi-agent arm and no evolutionary arm.** Replaced by graph-of-thoughts and MCTS at
  Checkpoint 4.0, before generation. A second hole.
- **Model scale is not tested.** Requires a second local model or hosted access; the primary claims
  do not depend on it.
- **`scripts/verify_corpus.py` still expects n = 120 per arm and exits 1 on G7.** The constant is
  stale deliberately: an expected sample size edited to match what was generated has the shape of the
  pathology this lab detects, even when innocent. Superseded by Amendment 3's table, not by a code
  change.

## Reproducing everything

```
python scripts/paradigm_audit.py                  # static + statistical, per arm and pooled
python scripts/run_paradigm_semantic.py           # semantic, all 2,495, resumable
python scripts/paradigm_calibrate.py              # determinism and knife-edge
python scripts/paradigm_stress.py --paths 1000    # tier-2 fragility
python scripts/paradigm_capacity.py               # constraint capacity, unfiltered panel
python scripts/paradigm_exposure.py               # deployed exposure
python scripts/paradigm_redundancy.py             # R(G) within arms
python scripts/paradigm_cross_arm_duplicates.py   # duplication across arms
python scripts/paradigm_redundancy_power.py       # power for the zeros
python scripts/paradigm_compute_matched_control.py  # RULE 11, the primary result
python scripts/paradigm_funnel.py                 # survival, equal-token and equal-n
python scripts/meta_analysis.py                   # everything above, in one artefact
```

The corpus itself is committed under `corpus/`, including every candidate's source, its draw record,
the hash-chained trial ledger, and its development and holdout return series. It cannot be
regenerated identically — the model's sampling was seeded but six arms interleaved across several
sittings were not — so the artefact is the record.
