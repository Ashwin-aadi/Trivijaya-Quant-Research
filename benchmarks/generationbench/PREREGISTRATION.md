# GenerationBench — pre-registration

**Committed before any strategy was generated. The git timestamp of this file is the evidence.**
Append-only. Anything analysed that does not appear below is **exploratory** and must be labelled as
such in the same sentence as its result, wherever it is reported.

This is the second confirmatory experiment in this lab, after the generator-validation addendum. It
inherits that study's discipline and one of its scars: the addendum's coverage audit found that its
first pass had put the arms through each benchmark's *headline* only, and five further measurements
had to be added afterwards. **Every measurement each benchmark makes is listed in §4 here, in
advance, for that reason.**

---

## 1. The question

P1 through P3 held the generator fixed and varied the scrutiny. The addendum held the scrutiny
fixed and varied the generator, and found the benchmarks' conclusions unchanged across three
frontier models. This study holds **both the model and the scrutiny fixed, and varies the
methodology by which strategies are generated.**

> **Does how an AI system is asked to generate a trading strategy change what survives an audit,
> a stress test and a capacity model — once the comparison is made at equal compute?**

The final clause is the study. A five-call committee that beats a one-call prompt has learned
nothing; it thought five times as hard. RULE 11 governs, and the control is plain prompting **run
k times and allowed to keep its best**, with k set by generated tokens.

## 2. The instrument, frozen

| | State | Last change |
|---|---|---|
| `src/audit/` | read-only permanently since P2's release | `61747ef`, 2026-07-28 |
| `src/stress/` | frozen | `7734fee`, 2026-08-02 |
| `src/capacity/` | frozen | `2c1b601`, 2026-08-03 |

Tagged `evalstack-p4`. AlphaAudit is at benchmark v1.3, RegimeStress at `regimestress-v1`, FlowState
at `flowstate-v1`. **A change to any of the three directories after this file is committed
invalidates this study and, under RULE 7, the holdout with it.**

The task specification is P1's frozen prompt, `src/generate/prompts.py`, digest
`f307433c7bda8595d52432b3bcb4f723663bfe706112a41f84e4beacfbde9934`. Every arm builds its prompt from
`build_prompt(theme_for(index))`. **Unlike the addendum, this digest matches P1's exactly**, so the
control arm and the treatment arms answer the identical task.

## 3. The arms

Model held fixed at `qwen2.5:7b-instruct-q4_K_M`, temperature 0.8, local via Ollama, per RULE 5.

| | Paradigm | Structure | Model calls per strategy |
|---|---|---|---|
| `G1` | Plain prompting | one call | 1 |
| `G2` | Chain of thought | one call, reasoning elicited first | 1 |
| `G4` | Planning | planner → signal → risk → implementer | 4 |
| `G5` | Reflection | draft → critique → rewrite | 3 |
| `G6` | Multi-agent | economist → statistician → quant → reviewer → implementer | 5 |
| `G7` | Evolutionary | population, scored, bred over generations | 12 candidates |

**`G3`, retrieval-augmented generation, is not run.** It needs a corpus of factor literature to
retrieve from and this repository holds none — `data/raw/` contains bhavcopy prices, calendars,
participant flows and the SEBI event timeline, and no embedding dependency is installed. Acquiring
and indexing a literature corpus is a project, not an arm, and the data budget is Rs 0. Dropped
here, before any data existed, rather than attempted badly. **This is a gap in coverage of the
paradigm space and is to be reported as one, not omitted.**

**`RQ5`, model scale, is not run.** It requires either a second local model or hosted access, and
the charter states the primary claims must not depend on it. They do not.

### The compute-matched control

The control is **not** G1 at its natural budget. For each treatment arm `T`:

1. Measure `tokens_per_accepted(T)` and `tokens_per_accepted(G1)` in **generated** tokens
   (`eval_count`), not prompt tokens and not calls. G2 makes one call and costs more than G1;
   a call-count ratio would hand it a free budget increase.
2. `k = ceil(tokens_per_accepted(T) / tokens_per_accepted(G1))`, rounded **up**, so the surplus
   goes to the control rather than the treatment.
3. Draw `k` consecutive indices from P1's 1,550-candidate corpus and keep the best by development
   Sharpe.

**Blocks are contiguous, never a uniform random sample.** P1's corpus is stratified —
`theme_for(index)` is `THEMES[index % 12]` — so `k` consecutive indices cover `k` consecutive themes
exactly as `k` fresh draws would. A uniform sample draws a random theme mixture with repeats: a
different sampling scheme, a different variance, and a confound invisible in the output.

**A block containing nothing rankable is a failed control draw and counts in the control's yield
denominator.** Only 225 of P1's 1,550 candidates executed and took a position, so for small `k` many
blocks are barren. Discarding them would compare the control's best blocks against every one of the
treatment's, inflating the control and manufacturing a null.

**`tokens_per_accepted(G1)` cannot be read from P1's corpus**, which was generated before anything
counted tokens. A live G1 arm is run for calibration only, at the same n as every other arm.

## 4. Every measurement, named in advance

**Generation**, per arm: yield `Y` (draws that execute and take a position), syntax-failure rate,
execution-failure rate, redundancy `R` (fraction inside an exact-duplicate cluster, by P2's
union-find over identical realised return series), near-duplicate pairs at r ≥ 0.9999, coverage `V`
(effective dimension of the span of factor exposures), generated tokens per accepted strategy,
wall-clock per accepted strategy.

**AlphaAudit**: static rejection rate and the class distribution of findings; semantic label
distribution; DSR at the ruled trial count; PBO; AUAP against random rejection at matched coverage,
for all seven layer combinations.

**RegimeStress**: fragility across regimes (the primary), fragility across CRR paths, knife-edge rate
under a 9e-15 panel perturbation, nondeterminism rate across hash seeds, fragility-predictor R² and
Spearman ρ out of population.

**FlowState**: binding constraint-based deployment capacity, capacity span, outflow/inflow capacity
ratio, alpha-decay half-life, mean Sharpe lost to costs, count profitable gross and unprofitable net.

## 5. Hypotheses

Each is stated so that a specific measurement falsifies it.

**H1 — Scaffolding raises yield before compute matching.** Every one of G2, G4, G5, G6, G7 has a
higher rankable rate than the live G1 arm at its natural budget. *Falsified if any treatment arm's
yield point estimate is at or below live G1's.*

**H2 — At matched compute, most of that advantage disappears.** At most two of the five treatment
arms exceed their own compute-matched control's yield at α = 0.01 (Bonferroni across five arms).
*Falsified if three or more clear it.*

**H3 — Reasoning depth trades against diversity.** Across the six arms, redundancy `R` correlates
positively with tokens per accepted strategy (Spearman ρ > 0), and coverage `V` correlates
negatively. *Falsified if either correlation has the opposite sign.*

**H4 — An honest trial counter erases the evolutionary arm's gains.** G7's best DSR, deflated at its
own candidate count (12 per returned strategy, not 1), is no higher than G1's best DSR deflated at
G1's. *Falsified if G7's best DSR exceeds G1's.*

**H5 — Nothing clears the statistical layer, in any arm.** Zero strategies across all arms reach
DSR ≥ 0.95 on development data. *Falsified by a single strategy clearing it.* This is a predicted
zero, stated as such: P1 admitted 0 of 1,550 and the addendum 0 of 60. **RQ2, full-stack survival,
therefore cannot carry a primary claim and is not asked to.**

**H6 — The static layer does not discriminate between paradigms.** The static rejection rate does not
differ across arms at α = 0.05. *Falsified if it does.* Note the confound stated in §7: this layer
raised one finding in the addendum's sixty strategies, and no human has confirmed it.

**H7 — No paradigm changes the published conclusions.** The AUAP null, the fragility-predictor null
and the capacity-constraint findings hold on every arm's population. *Falsified if any arm's AUAP
beats random rejection at matched coverage.*

## 6. Primary metric, test, and direction

**Primary: yield at matched compute.** For each treatment arm `T`, a two-proportion z-test of
`Y(T)` against `Y(control_T)`, two-sided, α = 0.05 Bonferroni-corrected to **0.01** across the five
treatment arms. Direction stated in advance: **H1 predicts the raw difference is positive; H2
predicts the matched difference is not, for at least three of the five arms.**

**Secondary, and reported whatever it shows:** every quantity in §4, per arm, with its sample size in
the same sentence.

### Sample size

**n = 120 draws per arm**, six arms, plus a resampled control per treatment arm.

The reason is power against the baseline this lab has already measured. At P1's rankable rate of
14.5% (225/1,550), α = 0.01 and 80% power:

| n per arm | Smallest detectable yield | Absolute difference |
|---|---|---|
| 60 | 42.3% | +27.8 pp |
| 100 | 35.2% | +20.7 pp |
| **120** | **33.2%** | **+18.7 pp** |
| 150 | 31.0% | +16.5 pp |
| 200 | 28.5% | +14.0 pp |
| 300 | 25.7% | +11.2 pp |

**n = 120 buys only large effects, and that is stated rather than hidden.** It cannot see a rise from
14.5% to 25%; it needs 167 draws per arm to see a rise to 30%. It is chosen because the effect this
lab has actually observed in generation is enormous — the addendum's frontier arms yielded 100%,
which needs n = 7 — and because compute is the binding constraint. **If every arm lands between 15%
and 30%, this study will be underpowered and must say so rather than report a null.**

### Generation cost estimate

From `reports/generation_throughput_estimate.md`: 18.1 s per strategy for a single call on the
RTX 4060 Laptop. Scaling by call count as an upper bound — intermediate stages are capped at 150–250
words and will generate far fewer tokens than a full strategy, so this over-states:

| Arm | Call-count equivalents | Estimated hours at n = 120 |
|---|---|---|
| G1 live | 1 | 0.6 |
| G2 | 2 | 1.2 |
| G4 | 4 | 2.4 |
| G5 | 3 | 1.8 |
| G6 | 5 | 3.0 |
| G7 | 12 | 7.2 |
| **Total** | **27** | **16.3** |

**This is an estimate, not a measurement, and the ten-generation probe at the start of Phase 4.1
replaces it.** The author's estimating record in this project is three over-estimates and no
under-estimates, so the true figure is expected to be lower. G7 additionally puts the backtester
inside the generation loop: 12 candidates × 11.8 s ≈ 142 s of CPU per draw, parallelisable, and
independent of the GPU budget.

## 7. Exclusion rules, fixed in advance

1. **Nothing is excluded for its result.** A draw that fails to parse, fails to execute, or takes no
   position is a datum and stays in the yield denominator.
2. **Every candidate is a trial**, including retries, discarded evolutionary individuals, and syntax
   failures. G7 is charged 12 per returned strategy.
3. **Nondeterministic strategies are retained and reported**, not dropped. P1 found 27 of 174
   survivors return a different Sharpe on rerun; `PYTHONHASHSEED` is pinned for this study and the
   rate is measured per arm as a paradigm property.
4. **The knife-edge exclusions P2 defined apply unchanged** at the stress stage, and the count
   excluded is reported per arm.
5. **No arm is dropped after its downstream results are seen.** Dropping an arm at Checkpoint 4.1,
   on yield and redundancy alone, is permitted and honest. Dropping one later is not.

## 8. Predictions, recorded before the data exists

**Mine.** G5 reflection wins on yield, because most P1 failures were conformance failures and a
critique pass targets exactly those. G6 multi-agent gains least per token, because four short role
prompts consume budget without touching the code contract. G7 wins before trial adjustment and loses
after it. Redundancy rises with scaffolding on every arm. **I expect H2 to hold: at most two arms
beat their matched control.** I expect H5's zero and H7's null to hold on every arm, because they
have held on 1,550 local strategies and 60 frontier ones.

**The PI's.** *To be recorded at Checkpoint 4.0, before generation, in the PI's own words.*

**If we are both wrong, the paper says so, in those words.**

## 9. The open decision this study cannot make for itself

**The trial-counter rule: pooled across arms, or one counter per arm?** Pooling punishes every
paradigm for the volume of the others; separating them makes cross-paradigm DSR incomparable. There
is no neutral choice, and CLAUDE.md requires the PI to make it in advance and in writing.

**Not decided here.** It is Checkpoint 4.0's first question, and the ruling is appended to this file
as an amendment **before any strategy is generated**. Precedent exists: the addendum's Amendment 2
resolved an equivalent ambiguity by publishing both readings and forbidding either alone.

## 10. Threats to validity, known before starting

1. **The semantic auditor's accuracy on this population is unmeasured.** P1's Cohen's κ was computed
   on the local corpus. Thirteen frontier strategies were flagged in the addendum and none was
   hand-labelled. Any per-arm semantic rejection rate inherits that uncertainty.
2. **The static layer may be inert rather than robust.** It raised one finding in sixty frontier
   strategies, and that finding carries no independent human confirmation — the PI declined to
   review it, logged as a waiver on 2026-08-04. If H6 holds and static rejection is flat across
   arms, "the layer does not discriminate" and "the paradigms do not differ" are **not
   distinguishable**, and the result must be reported as such.
3. **The paradigms are one implementation each.** G6's four personas are a prompt, not four models.
   A different implementation of the same idea could perform differently, and this study measures
   these implementations on this date.
4. **n = 120 sees only large effects**, per §6.
5. **The control reuses P1's corpus.** Its draws are already counted in AlphaAudit's ledger at
   N = 1,887. Both readings — control draws counted afresh, and control draws treated as
   already-counted — are published, and neither is quoted alone.
6. **G3 is missing**, so the paradigm space is covered with a hole in it.

---

# AMENDMENT 1 — Two arms swapped, the trial-counter rule settled, and the PI's predictions

**Committed 2026-08-04, before any strategy was generated.** Nothing in the corpus directory existed
when this was written; `benchmarks/generationbench/corpus/` was created empty by the commit that
carries this amendment, and its emptiness at that commit is the evidence.

This amendment records the PI's rulings at Checkpoint 4.0 and the design changes they authorised.
Everything above this line stands as written; nothing has been edited.

## 1.1 — The arms: G6 and G7 replaced

| | Was | Is now | Structure |
|---|---|---|---|
| `G6` | Multi-agent (chain of four roles, then an implementer) | **Graph of thoughts** | 3 independent proposals → 2 aggregations under different criteria → 1 merge → implement. **7 calls.** |
| `G7` | Evolutionary (4 population × 3 generations = 12 candidates) | **Monte Carlo tree search** | 12 iterations, each one expansion call plus one completion call, over a 3-layer design tree. **12 candidates, 24 calls.** |

**Both retired modules remain on disk** at `src/generate/paradigms/multi_agent.py` and
`evolutionary.py`, still tested and still under the prompt-circularity check. They are not arms.

**Why the swap is admissible.** RULE 10 requires the analysis to be written before the data. No data
exists: not one model call has been made for P4. §7 rule 5 of this file permits dropping an arm
before its downstream results are seen and forbids it afterwards; this is the former, by a margin of
the entire experiment. The swap is recorded here rather than by editing §3 so that the original
design is not overwritten.

**Why these two.** The retired G6 was a *chain*: each role saw only its predecessor's output and
could therefore only elaborate it. Graph of thoughts aggregates *independent* proposals — an
operation a chain cannot express — which makes it a genuinely different point in the paradigm space
rather than a longer version of G4 planning. The retired G7 and its replacement both close a loop
with a fitness signal, which is the property H4 and RQ4 depend on; what MCTS adds is a budget linear
in one parameter, where `population × generations` is a product of two that interact.

**What this costs in coverage, stated rather than buried.** P4 no longer measures a multi-agent
paradigm or an evolutionary one. Both are well represented in the literature, and their absence is a
second hole in the paradigm space alongside G3's. This is to be reported in the paper's threats
section, not in a footnote.

### H4, restated for the arm that now exists

> **H4 — An honest trial counter erases the search arm's gains.** G7's best DSR, deflated at its own
> candidate count (12 per returned strategy, not 1), is no higher than G1's best DSR deflated at
> G1's. *Falsified if G7's best DSR exceeds G1's.*

The hypothesis is unchanged in substance: it was never about breeding specifically, but about
whether *iterating with a fitness signal* survives honest accounting. Only the arm's name changes.

### The revised cost estimate

Two figures, because the call-count scaling used in §6 over-states arms whose intermediate stages
are word-capped, and by different amounts now that two arms have changed shape.

| Arm | Calls per draw | Call-count ceiling (h) | Structure-aware estimate (h) |
|---|---|---|---|
| G1 live | 1 | 0.6 | 0.6 |
| G2 | 1 | 1.2 | 1.2 |
| G4 | 4 | 2.4 | 1.9 |
| G5 | 3 | 1.8 | 1.6 |
| G6 graph of thoughts | 7 | 4.2 | 1.9 |
| G7 MCTS | 24 | 14.5 | 9.4 |
| **Total at n = 120** | | **24.7** | **16.6** |

The structure-aware column charges a word-capped intermediate stage at 0.3 of a full strategy
generation and a full implementation at 1.0. **Both columns are estimates and neither is a
measurement.** `scripts/probe_paradigms.py` replaces them before the run, per CLAUDE.md's Phase 4.1
halt, and the projection it produces is what the run is committed against.

G7 additionally puts the backtester inside the generation loop — 12 backtests per draw at roughly
11.8 s each, about 3.9 h of CPU across the arm, parallelisable and independent of the GPU budget.

## 1.2 — The trial-counter rule (PI ruling, resolving §9)

> **Publish both per-arm and pooled DSR, with per-arm treated as the primary analysis.** Per-arm
> measures the methodology fairly, because each paradigm has a different search budget. Pooled
> remains valuable as a sensitivity analysis and for comparison with the entire experiment. Report
> both explicitly and never rely on only one.

**Binding consequences.**

1. Each arm writes its own hash-chained ledger at
   `benchmarks/generationbench/corpus/<arm>/trial_ledger.jsonl`. The pooled count is the sum of the
   six and is computed at analysis time, never maintained as a seventh file that could drift.
2. **P1's project ledger is not touched.** It stands at the count AlphaAudit's published results
   were deflated against; appending P4's draws would retrospectively change a released paper's N.
3. **No table, figure or sentence may report one reading without the other**, matching the addendum's
   Amendment 2. A per-arm DSR quoted alone overstates; a pooled DSR quoted alone punishes every arm
   for G7's volume.
4. The ledger increments by **candidates evaluated**, not by draws: 24 attempts for a 12-iteration
   MCTS draw that retried nothing, 1 for a plain draw that conformed first time.

## 1.3 — The search arm's fitness objective (PI ruling, resolving Checkpoint 4.0 question 4)

> **Use net-of-costs performance as the optimisation objective.** Every benchmark in this research
> programme ultimately evaluates net performance after realistic transaction costs, so the search
> objective should remain aligned with the benchmark rather than optimise gross returns.

Implemented in `src/generate/paradigms/fitness.py` as development-window Sharpe net of the Phase 1.1
Indian cost model. **The confound this creates is pre-registered here:** G7 is the only arm that
receives cost information during generation, so part of any advantage it shows may be
cost-optimisation rather than search quality. Any G7 result must be reported with that sentence
attached.

## 1.4 — Sample size (PI ruling, resolving Checkpoint 4.0 question 3)

> **n = 120 per arm.** Six paradigms × 120 = 720 generated strategies — already several times larger
> than the frontier-model validation. Additional compute beyond 120 gives diminishing returns
> relative to the extra GPU time.

Unchanged from §6, and §6's power table and its explicit statement that n = 120 cannot see a rise to
25% stand exactly as written.

## 1.5 — The frozen stack (PI ruling, resolving Checkpoint 4.0 question 5)

> **Approved.** AlphaAudit v1.3, RegimeStress v1.0 and FlowState v1.0 are permanently frozen for
> Project 4. No benchmark logic, thresholds or scoring rules may change after generation begins. Any
> future benchmark improvements belong to later versions and must never alter Project 4 results.

## 1.6 — Compute efficiency as an auxiliary metric (PI addition)

> **Compute efficiency will be reported as an auxiliary metric** (rankable strategies per million
> generated tokens, and per GPU-hour). **This metric is exploratory and will not be treated as a
> primary hypothesis or influence confirmatory statistical conclusions.**

Added because MCTS will consume substantially more compute than reflection or chain of thought, and
a reader will ask whether the extra cost is justified. It is pre-registered *as exploratory*, which
is the distinction RULE 10 exists to preserve: it is declared in advance and it still cannot support
a confirmatory claim. Every report of it carries the word "exploratory" in the same sentence.

## 1.7 — The PI's predictions, recorded before the data exists (resolving §8)

Verbatim, in the PI's own words:

> My pre-registered prediction is as follows. Plain prompting will produce the weakest overall
> results. Chain-of-Thought and Planning will provide modest improvements over the baseline.
> Reflection will offer the best balance between quality and computational cost. Graph-of-Thoughts
> will improve diversity through reasoning aggregation but is not expected to substantially
> outperform Reflection. Monte Carlo Tree Search will achieve the highest raw optimisation
> performance before trial correction, but after Deflated Sharpe adjustment its advantage will
> diminish because of its substantially larger search budget. Overall I predict Reflection will be
> the strongest practical methodology under a fixed compute budget.

**Where the two of us agree:** reflection wins on practical grounds; the search arm leads before
trial adjustment and gives it back after.

**Where we differ, and it is worth recording because only one of us can be right:** the PI expects
graph of thoughts to *raise* diversity through aggregation. §5's H3 predicts the opposite — that
redundancy rises and coverage falls as scaffolding deepens, on every arm. **H3 is the pre-registered
hypothesis and it is not amended.** If G6's coverage rises, H3 is falsified and the PI was right;
the paper reports that in these words.

**If we are both wrong, the paper says so, in both sets of words.**

---

# AMENDMENT 2 — Equal compute budget per arm, replacing equal n

**Committed 2026-08-05, before any draw beyond the ones listed below.** PI ruling, adopted against
my stated reservations, which are recorded here rather than omitted.

## 2.1 — The change

Fixed `n = 120` per arm is replaced by **fixed generated-token budget per arm**. The research
question becomes: *under an equal compute budget, which reasoning methodology produces the most
surviving strategies?*

**Budget: 439,200 generated tokens per arm**, set by the search arm at n = 60.

## 2.2 — Sample sizes, from the probe of 2026-08-05

Measured output tokens per draw (2 draws each): G1 448, G2 388, G4 943, G5 1404, G6 2514.
**G7 is estimated at 7,320 and is not yet measured** — if the measurement differs, every other arm's
n moves with it and this table is reissued before those draws.

| Arm | Tokens/draw | **n** | Est. hours |
|---|---|---|---|
| G1 | 448 | 980 | 3.2 |
| G2 | 388 | 1130 | 3.3 |
| G4 | 943 | 465 | 3.8 |
| G5 | 1404 | 315 | 3.2 |
| G6 | 2514 | 175 | 3.2 |
| G7 | ~7320 | 60 | 4.0 |

Total ≈ 20.7 h.

## 2.3 — What this costs, stated in advance

1. **This design change was made after generation began**, with the first draws of five arms
   visible. It is therefore **exploratory as to the choice of design**, and every report of the
   equal-budget result must say so in the same sentence. The hypotheses H1–H7 are unchanged and
   remain confirmatory.
2. **Equal compute is not equal precision.** G7 at n = 60 detects only very large effects; G2 at
   n = 1130 detects small ones. **A null from G7 is uninformative and must not be reported as
   evidence of no effect.**
3. **The primary metric changes from a rate to a count.** Yield per draw and surviving strategies
   per budget can disagree — a paradigm can have the better rate and the worse count. **Both are
   reported, neither alone.**
4. **G7 cannot shrink below what has already been drawn.** If more than 60 exist when this takes
   effect, all are kept and the real number is reported.
5. Matching is on **generated tokens**, per RULE 11. Wall-clock hours are reported alongside and are
   not the matching unit — G7's hours include backtest CPU that is not generation.

## 2.4 — Unchanged

Every hypothesis, exclusion rule, measurement in §4, the frozen stack, the trial-counter rule of
Amendment 1.2, and the compute-matched best-of-k control against P1's corpus. The control's k is now
computed against each arm's measured tokens per accepted strategy, exactly as §3 already specifies.

---

# Amendment 3 — the realised design, recorded before any Phase 4.2 result is read

**Committed 2026-08-07, on the PI's ruling at Checkpoint 4.1 question 5.** Amendment 2 switched the
design from equal-n to equal-token but was written while generation was still running, so it carries
the *intent* and an estimated 439,200-token budget. This records what was actually spent. No
hypothesis, exclusion rule or measurement changes; §2.3's four costs stand unaltered.

## 3.1 — What was generated

| arm | paradigm | draws | output tokens | tokens/draw | model calls | trials |
|---|---|---|---|---|---|---|
| G1 | plain | 830 | 407,588 | 491 | 844 | 844 |
| G2 | chain-of-thought | 800 | 408,215 | 510 | 811 | 811 |
| G4 | planning | 365 | 409,161 | 1,121 | 1,465 | 370 |
| G5 | reflection | 288 | 408,324 | 1,418 | 870 | 582 |
| G6 | graph-of-thoughts | 152 | 410,177 | 2,699 | 1,068 | 156 |
| G7 | MCTS | 60 | 408,053 | 6,801 | 1,436 | 736 |

**Matched to 0.6%** — a spread of 2,589 output tokens on a base of ~408,000. This is the figure
RULE 11 requires and the one the primary comparison rests on. Pooled trial count **3,499**.

Realised wall-clock 19.64 h against the 16.6 h structure-aware projection (+18%), inside the 24.7 h
call-count ceiling. Accepted by the PI at Checkpoint 4.1 question 2.

`scripts/verify_corpus.py` still expects n = 120 per arm and therefore exits 1 on G7. The constant
is left stale deliberately: an expected sample size edited to match what was generated has the shape
of the pathology this lab detects, even when innocent. It is superseded by this table, not by a code
change.

## 3.2 — Fragility is measured at Tier 2 only, and is not comparable to P2's

**PI ruling, 2026-08-07, with the cost in front of them.** P2 measured tier agreement on mean
performance at Spearman 0.897 and on **fragility at 0.620** (n = 125), and published the conclusion
that *"Tier 2 cannot be substituted for Tier 1 when fragility is the quantity of interest."*

Tier 1 is nonetheless unavailable: P2 spent **124.5 CPU-hours** on 125 strategies, so P4's 315 would
cost upwards of 250 on consumer hardware at a zero budget. Tier 2 at 1,000 paths costs seconds.

**The justification is affordability, not agreement.** No sentence in the paper may claim or imply
that the tiers agree on fragility; P2 is the paper that measured they do not. Two consequences bind:

1. P4's fragility figures are **not interchangeable with P2's** and are never quoted alongside them
   as though continuous.
2. Tier 2 is a noisy proxy for fragility, so power to separate arms is reduced. **A null between
   arms means "not detected", never "not there"** — the same caution Amendment 2 §2.3(2) attaches to
   G7 at n = 60.

The cross-arm comparison remains internally valid: all six arms are measured by the same tier, block
length, seed and rule, and a shared bias cancels when ranking arms rather than reporting a level.
The caveat is emitted into every arm's `fragility.json` so it travels with the numbers.

## 3.3 — Semantic audit covers all draws, not only those that traded

Deciding this before results are read, per RULE 10. The semantic layer needs source text alone, so
it is run over **all 2,495 candidates** rather than the 315 that took a position. RQ3's leak-class
distribution is then measured over what each paradigm actually emits; restricting it to traders
would condition the distribution on having traded, which is a selected sample. The cost of the wider
population is 2.6 h of GPU against a stage that is not the critical path.
