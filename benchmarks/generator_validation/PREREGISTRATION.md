# PRE-REGISTRATION — Are three frozen benchmarks robust to the generator?

**Committed before any strategy is generated. This file is append-only; its git timestamp is the
evidence.** Any analysis performed that does not appear below is **exploratory** and must be
labelled as such in the same sentence as its result.

**Status: DRAFT AWAITING PI SIGN-OFF.** Two blanks marked **[PI RULING REQUIRED]** must be filled
before a single strategy is generated. Generating first would void this document's purpose.

---

## 1. The question

Every number published in AlphaAudit, RegimeStress and FlowState was measured on strategies written
by one local 7B model. The obvious objection is that the three benchmarks characterise that model
rather than machine-written strategy research.

> **Do frontier language models materially change the behaviour or conclusions of the three frozen
> benchmarks?**

The benchmark suite is the fixed instrument. The generators are the experimental subjects.

## 2. What this study is, and is not

**It is a measurement of realistic frontier usage.** Strategies are obtained the way a working
researcher would obtain them: a single request to a chat interface asking for five strategies in
one Markdown file. This is a deliberate PI ruling, taken with the alternatives on the table.

**It is not a compute-matched comparison, and no arm will be reported as a win.** RULE 11 requires
comparison at equal token budget. A chat interface reports no token counts and applies undisclosed
amounts of internal reasoning, so the control RULE 11 demands **cannot be constructed**. This study
therefore reports rates and distributions descriptively. Any sentence claiming one generator is
better than another is out of scope and may not appear in the write-up.

**It is not a replacement for the local-model result.** The scientific claims of P1–P3 rest on the
local arm, where sample size is large, generation is seeded, temperature is fixed, and every draw is
independent. This arm is a robustness check on those claims, reported with its limitations stated in
the abstract rather than in a footnote.

**The prompt is not P1's prompt.** It carries P1's interface contract verbatim but is re-wrapped to
request five strategies in one file. The digests therefore differ, and both are recorded in
[`PROMPT.md`](PROMPT.md). The envelope is a known difference between the arms and is not a
confound that can be removed — it is the realistic-usage condition being studied.

## 3. Arms

| | Generator | Interface | Role |
|---|---|---|---|
| `M₀` | `qwen2.5:7b-instruct-q4_K_M`, local, temp 0.8, seed 42+i | API, one strategy per call | reference, already collected |
| `M₁` | Claude | chat, five strategies per request | frontier |
| `M₂` | Gemini | chat, five strategies per request | frontier |
| `M₃` | GPT | chat, five strategies per request | frontier |

Product versions and access dates are recorded at generation time and are part of the result: this
measures three products on specific dates, not three laboratories.

**Collection conditions, fixed in advance.** Each request is issued in a new conversation with
memory, personalisation, custom instructions and project context disabled. The raw response is saved
verbatim and hashed before any parsing. No response is regenerated, edited, retried or discarded for
any reason — including refusal, truncation or malformed output. A failed response is a datum.

## 4. Sample size

**n = 5 per frontier arm, 15 frontier strategies total**, against M₀'s 1,550 draws.

The reason is that this is one realistic request, which is the exposure being measured. The
consequence is stated plainly: **at n = 5, a rate of 3/5 carries a 95% confidence interval of
roughly 15% to 95%.** This arm can therefore support descriptive and qualitative claims and cannot
support a statistical comparison of rates. The write-up must say so wherever a rate appears.

**If additional requests are made, the total per arm is declared here by amendment before any
additional generation, and every draw collected is analysed.** Collecting until a result appears,
or discarding a chat that produced poor strategies, would convert this into the pathology the lab
exists to detect.

## 5. Hypotheses

Each is stated so that a specific measurement falsifies it. Reference values are M₀'s, from
`benchmarks/alphaaudit/RESULTS.md`.

**H1 — Executability rises sharply.** M₀'s rankable rate (executes *and* takes a position) was
**14.5%**. Predicted frontier rankable rate **≥ 40%**. *Falsified if* the pooled frontier rate is
below 25%.

**H2 — Audit pass rate does not improve.** M₀ static-flagged **14.3%** of candidates. Predicted
frontier flag rate **within ±10 points of that, and not lower**. *Falsified if* the pooled frontier
flag rate is below 5%, which would indicate frontier models avoid the leakage classes M₀ produced.

**H3 — The `full_sample_statistic` blind spot is an instrument property, not a corpus property.**
M₀ produced **zero** flags in this class across 1,550 candidates, which P1 published as a suspected
blind spot rather than a clean result. Predicted: **zero again** across the frontier arms.
*Falsified if* any frontier strategy is flagged in this class — which would be the more useful
outcome, since it would locate the blind spot in the corpus rather than the detector.

**H4 — Diversity does not improve, and may fall.** Better instruction-following is expected to
converge on canonical implementations of each named theme. Predicted: frontier exact- or
near-duplicate rate **≥ 0**, with at least one within-arm near-duplicate pair (r ≥ 0.9999) across
the three arms. *Falsified if* zero near-duplicate pairs appear in any arm.

**H5 — Downstream benchmark outputs are invariant to the generator.** Capacity and fragility are
properties of Indian market liquidity, not of who wrote the code. Predicted: frontier median
binding capacity **within 2× of the reference corpus median of ₹2.60 crore**. *Falsified if* it
differs by more than 2× in either direction.

**H6 — The published conclusions do not move.** M₀ produced **0 of 631** strategies clearing
deflation at an honest trial count. Predicted: **zero frontier strategies clear deflation** under
whichever trial-counter rule §7 fixes. *Falsified if* any does.

**The prediction, stated as a whole:** executability rises a great deal, and nothing downstream of
executability moves. If that holds, the three benchmarks measure machine-written strategies rather
than one local model, which is the finding that licenses anyone else using them.

## 6. Primary metric and analysis

**Primary metric:** the rankable rate per arm — the fraction of generated strategies that execute
over 2020–2024 and take at least one position.

**Comparison:** each frontier arm against M₀'s 225/1,550, reported as a proportion with an exact
(Clopper–Pearson) 95% interval. **Direction: frontier greater than local.** Given §4, the interval
is reported for honesty about precision, not to support a significance claim.

**Secondary, all pre-specified:** static flag rate and its class distribution; near-duplicate rate;
deflated Sharpe at the §7 trial count; fragility score distribution; binding deployment capacity
distribution; and the count of strategies carrying all three benchmark verdicts.

**Every stage runs at its frozen parameters.** No parameter of `src/audit/`, `src/stress/` or
`src/capacity/` may change as a result of anything this study shows. If the frontier corpus exposes
a defect — as the local corpus did for FlowState — that is a finding to report and, per the PI
ruling of 2026-08-03, a repair to an objective omission may be made and logged in the relevant
`CORRECTIONS.md`. A change that adapts a threshold to the results may not.

## 7. **[PI RULING REQUIRED]** The trial counter

Deflated Sharpe requires an honest *N*, and there is no neutral choice here:

- **Pooled**, at cumulative *N* across every arm ever run. M₀ contributed 1,887 trials, so a
  frontier strategy would be deflated against a denominator overwhelmingly earned by another
  generator. Defensible as "this lab has tried this many things," punitive as a comparison.
- **Per-arm**, each frontier arm deflated at its own small *N*. Makes each arm internally honest and
  makes cross-arm DSR incomparable, since a low-*N* arm faces a much lower bar.

**This must be decided before generation and cannot be decided after seeing results.** My
recommendation is **per-arm, with the pooled figure reported alongside it** and the incomparability
stated — but the ruling is the PI's.

## 8. Exclusion rules, fixed in advance

1. **Nothing is excluded for performance.** No strategy is dropped because its result is
   uninteresting, implausible or disappointing, in any arm, at any stage.
2. **Unparseable output is a datum**, counted in the denominator as a syntax failure, never dropped.
3. **Runtime failure is a datum**, counted and classified by exception type as in P1.
4. **Nondeterministic strategies are excluded from the stress suite only**, by the frozen P2 rule,
   and reported as a count with their worst Sharpe swing.
5. **Knife-edge strategies are excluded from primary fragility statistics only**, by the frozen P2
   rule, and reported as a count.
6. **A strategy that fails an earlier stage does not reach a later one**, which is the funnel P1–P3
   already define; the count lost at each stage is reported per arm.

## 9. **[PI RULING REQUIRED]** Holdout

The PI has authorised evaluation on the grounds that the frozen benchmarks and their published
results are untouched, and that each generator is a new subject examined by an unchanged apparatus.
RULE 7's three conditions:

| Condition | Status |
|---|---|
| 1. Evaluation apparatus unchanged | **Holds.** `regimestress-v1` and `flowstate-v1` are tagged; each generator is one new subject. |
| 2. Auditor never modified after P2's release | **Holds.** `src/audit/` untouched since. |
| 3. Pre-registered under RULE 10 before any generator runs | **This file, once signed off and committed before generation.** |

**What remains for the PI to fix here is scope:** which measurements the holdout is used for. My
recommendation is that H1–H5 are answered entirely on development data and the holdout is spent
only on H6, since H6 is the only hypothesis about whether a *published conclusion* moves. Spending
it on the rest buys nothing and cannot be undone.

Every access is logged in `DECISIONS.md` with timestamp and authorisation. After evaluation, no
tuning of anything follows.

## 10. What would make this study worthless

Recorded so it can be checked against afterwards:

- A chat regenerated, retried or discarded because its output was poor.
- Additional requests issued after seeing results, without amending §4 first.
- Any frozen benchmark parameter changed because of what this corpus showed.
- An arm reported as a win despite §2.
- A rate reported without its sample size.
- Any analysis in the write-up not appearing above, presented as confirmatory.

---

## AMENDMENT 1 — sequential collection, 2026-08-03, before any strategy was generated

Arms are collected **one at a time**, beginning with GPT, rather than all three at once. This is a
scheduling decision and changes nothing about §4's committed sample size.

**The commitment it requires, made here before any result is seen:** all three arms will be
collected and analysed regardless of what any earlier arm shows. Stopping after an arm that produced
a striking result, or adding arms until one does, is optional stopping and would void this
pre-registration. If an arm is ever *not* collected, the reason is recorded here and the study is
reported as one-armed or two-armed rather than silently trimmed.

**Deferred, and permitted to be deferred because they bear on no measurement taken today:** the
§7 trial-counter ruling and the §9 holdout scope. Consequently **no Deflated Sharpe Ratio is
computed and the holdout is not touched** until both are ruled. H6 is therefore untestable today and
remains open. H1–H5 are answered entirely on development data and need neither ruling.

---

## AMENDMENT 2 — the two open rulings, settled 2026-08-03, before any strategy was generated

Both blanks in §7 and §9 are now filled by PI ruling. The draft status at the head of this file is
discharged: **no [PI RULING REQUIRED] blank remains, and generation is cleared to begin.**

### §7 settled — per-arm deflation, plus a matched-N comparison

> **PI, 2026-08-03:** *"Use per-arm DSR for each generator, plus the matched-N comparison against
> M₀ using random subsampling repeated with a fixed seed. Report both."*

**Primary.** Each arm is deflated at its own trial count. For the GPT arm that is N = 5.

**The comparison, which is what H6 is actually tested on.** Deflating GPT at N = 5 and M₀ at
N = 1,887 compares two different search intensities, not two generators. The hurdles are not close:
computed before any frontier strategy existed, a raw Sharpe of **0.648** clears DSR ≥ 0.95 at N = 5,
against **1.791** at N = 1,887. A strategy landing between those two numbers clears under one rule
and fails under the other, which is precisely why this was fixed in advance.

The matched comparison, fully specified so it cannot drift:

1. Draw a random subsample of M₀'s **rankable** strategies — those that executed *and* took a
   position, n = 225 — of the same size as the frontier arm under comparison.
2. Deflate every drawn strategy at the **same N as the frontier arm**, not at 1,887.
3. Record whether at least one strategy in the subsample reaches DSR ≥ 0.95.
4. Repeat **1,000 times at seed 42**, fixed here and not re-drawn.
5. The empirical p-value is the fraction of subsamples reaching that bar. The frontier arm's own
   result is read against that distribution.

Rankable is the sampling frame because a strategy that never traded has no return series to deflate.

**Both figures are reported wherever either appears**, and no sentence may quote the per-arm DSR
without the matched-N result beside it.

**H6 is hereby restated** in terms of the settled rule, without changing its content or direction:
*no frontier strategy clears DSR ≥ 0.95 at its own arm's N, and the frontier arm does not clear the
matched-N comparison more often than M₀ subsamples do.* The matched comparison is primary.
*Falsified if* either fails.

### §9 settled — the holdout is reserved for H6 and spent once

> **PI, 2026-08-03:** *"Reserve it only for H6, evaluate it once after GPT, Claude and Gemini have
> all been collected, and never use it during development or methodology decisions."*

Binding consequences, all of which constrain work already scheduled:

1. **One evaluation for the whole study**, after all three arms are collected — not one per arm.
   RULE 7's amendment would have permitted one per subject; the PI has elected to spend less than
   the rule allows.
2. **H1 through H5 are answered entirely on development data** and no holdout figure may appear in
   any of them.
3. **The holdout informs no development or methodology decision**, including any choice about
   parsing, exclusion, stage ordering or presentation. Any such decision made after the evaluation
   is contamination and must be reported as such.
4. **After the evaluation, no tuning of anything follows.** If the result is disappointing, that is
   the result.
5. The evaluation is logged in `DECISIONS.md` with its timestamp and this authorisation.

---

## AMENDMENT 3 — n raised to 20 per arm, 2026-08-03, before any reply was parsed

§4 fixed n = 5 per arm and required any increase to be declared here **before the additional
generation**. The PI has elected four requests per arm rather than one.

**n = 20 per frontier arm**, as four independent requests of five strategies each.

**The timing, which is the only thing that makes this legitimate.** The first reply
(`gpt_001.md`) had been received, saved and hashed when the decision was taken, but **nothing had
been parsed**: no parse rate, no audit verdict, no backtest, no rate of any kind existed. The
increase was therefore chosen against the width of a confidence interval, not against a result. Had
the extractor been run first, this amendment would not have been available, and the arm would have
stood at five.

At n = 20 an observed 12/20 carries a 95% interval of roughly 36–81%, against roughly 15–95% for
3/5. Still wide. It is the difference between a rate that can be written down and one the interval
swallows whole.

**Applies to every arm.** Claude and Gemini are collected at four requests each, so the arms remain
comparable. If an arm is collected at a different size the reason is recorded here.

**Within-request non-independence now has a second layer.** Five strategies inside one reply are
not independent of each other, as `PROMPT.md` records. Four replies are independent of each other.
The arm is therefore **four independent clusters of five**, not twenty independent draws, and any
interval computed as though n = 20 were independent will be too narrow. Where a clustered
calculation is not performed, the figure is reported with this caveat attached.

### Raw replies, hashed before parsing

| File | Bytes | SHA256 |
|---|---|---|
| `gpt_001.md` | 11,154 | `c06c7ebc099b1ad2c54e403a5fe8450f9f7d902ac074ed6893dd5aeac3e6a5ea` |
| `gpt_002.md` | 10,738 | `8861f3880ec0c5a75cbffb70bba804f624280576655ea9265efa0e78022eac58` |
| `gpt_003.md` | 11,323 | `9b2c1f385a45c051f42ae92aed86e1082d69672b82dcf163253f4c0971646609` |
| `gpt_004.md` | 10,578 | `862eae10c9913e623ceb4beaeb95b8d5d8474bce3cfc1962926cee588bf8d385` |

**Provenance limitation, recorded because it cannot be repaired retrospectively.** All four replies
reached the repository by being pasted into a conversation rather than saved directly from the
browser. The content was transcribed faithfully, but chat transport can normalise whitespace and
unicode, so these bytes are **not certified identical to what the product emitted**. The model is
recorded as the PI reported it --- *"ChatGPT base core model"* --- which is not a version string and
does not pin the arm to a specific model release.

---

## AMENDMENT 3 — model identity, the N reading, and full publication

Recorded 2026-08-04, **after all three arms were collected and their development-window results
were seen.** That timing is stated plainly: nothing below alters a hypothesis, a threshold or a
metric, and the one open question it settles was raised in writing before any arm's result was
known. But it is an amendment made with results in hand, and a reader is entitled to weigh it as
such.

### The subjects, as the PI identified them

| Arm | Model, as the interface reported it |
|---|---|
| `gpt` | GPT, base model |
| `claude` | Claude Opus, high effort |
| `gemini` | Gemini Pro |

None of these is a version string, and none pins an arm to a specific model release. The arms are
reproducible as *procedures* and not as *subjects*: re-running this study in six months would issue
the same prompt to different weights under the same product names. This is a genuine limitation of
collecting frontier arms through consumer chat interfaces, and it is not fixable within the ₹0
budget that RULE 5 and the market-data ruling impose.

The `claude` arm additionally carries the transport caveat already recorded for `gpt`: replies
reached the repository by being pasted into a conversation. The `gemini` arm carries it too.

### §7's N, settled by reporting both

> **PI, 2026-08-04:** *"the amendment fork do what you feel like i think we should report both"*

Amendment 2 fixed per-arm deflation and illustrated it with **N = 5**, the arm size expected when it
was written; the arms as collected are four requests of five, so *"its own trial count"* now reads
**N = 20**. Both are computed and **both are published in every table where either appears**. No
choice between them is made, because the choice would change nothing: **0 of 60 frontier strategies
clear DSR ≥ 0.95 under either reading**, and the matched-M₀ comparison rejects at both.

Had the two readings disagreed, this amendment would have had to choose, and choosing after seeing
results would have been indefensible. They do not disagree, and that is the only reason reporting
both is adequate rather than evasive.

### Publication scope

> **PI, 2026-08-04:** *"just publish all the results dont hide anyting whther negative positive
> bell or above the threshold"*

`RESULTS.md` is generated from run artifacts by a single script and contains every measured
quantity, including:

- **H2 falsified** — the static auditor raised one finding across 60 frontier strategies.
- **H3 not supported** — that finding was `snooped_parameter`, not the predicted class.
- **H5 falsified on one arm** — Gemini Pro's median deployment capacity is **0.31×** the local
  corpus, outside the pre-registered 2× band, in the opposite direction from the one being watched.
- The auditor layers that flagged **nothing**, reported as prominently as the ones that did.

No result is omitted, and no failed hypothesis is relegated to a limitation.
