# Write-up notes for P4 — PI instructions, 2026-08-07

Recorded at the PI's request during the Checkpoint 4.1 halt, before Phase 4.2 has produced a single
number. **Nothing here is a finding.** Items marked PREDICTION are pre-registered expectations under
RULE 10 and must be labelled as such in the paper if they turn out right; items marked FRAMING are
instructions about how results are to be presented once they exist.

## 1. FRAMING — the paper must not be plain-prompting-centric

P1's headline — *the dominant failure mode is generated code that does not run* — was measured on a
local 7B under plain prompting and one generation method only. It was stated as a property of
machine-written strategies. P4 has six methods at equal token budget, so that claim is now testable
rather than assumed, and **every headline number must be recomputed per paradigm, not inherited from
P1 and generalised.**

Concretely, every quantity P1 and P2 and P3 reported for the 1,550-strategy corpus is to be reported
for **all six arms**: yield, leak-class distribution, static and semantic audit pass rates, DSR and
PBO at each arm's honest trial count, fragility, deployment capacity, alpha decay, duplicate
clustering, and the holdout evaluation. No arm is the reference case in the prose. G1 is one row.

## 2. FRAMING — do not understate MCTS

G7 is to be reported at whatever it actually achieves, in both directions and in the same place.
Its 58.3% development yield is real and is also an artefact of optimising the metric (see
`reports/checkpoint_4.1.md` §6); both facts go in the same paragraph, not one in Results and the
other in Limitations. If it wins after deflation at 736 trials, that is the result. If deflation
erases it, that is RQ4's answer and equally the result.

## 3. ESTABLISHED — "front end improves, back end lags" is already measured, not predicted

I initially logged this as a prediction awaiting P4. That was wrong. The PI corrected it and the
evidence is in `benchmarks/generator_validation/`, which ran before P4 on hand-collected transcripts
at ₹0: GPT, Claude Opus and Gemini Pro, four requests of five strategies each, 60 strategies, through
the identical frozen funnel, pre-registered with six hypotheses, holdout authorised 2026-08-04.

The front half improves completely and the back half does not move at all:

| | Local M0 | GPT | Claude | Gemini |
|---|---|---|---|---|
| Executed and took a position | 14.5% | 100% | 100% | 100% |
| Clearing DSR ≥ 0.95, holdout | — | 0/20 | 0/20 | 0/20 |
| Holdout Sharpe, mean | −1.0351 | −0.9491 | −0.5781 | −1.4768 |
| Knife-edge under a 9e-15 panel change | 19.9% | 3/20 | 7/20 | 4/20 |
| Duplicates across independent requests | 11 clusters | 4 | 3 | 2 |

H1 confirmed 3 of 3, H6 confirmed 3 of 3 on both halves, H4 confirmed 3 of 3. **This is a
confirmatory, pre-registered, holdout-verified result and the paper states it as one.**

The one place to be careful: H2 was *falsified* — the static layer raised **1 finding in 60**, so
"audit pass rate does not improve" did not hold as written. That study reports the ambiguity rather
than resolving it: a layer that returns the same verdict regardless of author is either robust or
inert, and 1 finding in 60 cannot separate them. **Do not launder that into "frontier code is
clean."** It is RQ4 and it is unresolved.

## 4. SCOPE — what the two studies together do and do not license

Both axes are now measured, and they are orthogonal:

* **generator_validation** — vary the *model*, hold methodology fixed at plain prompting.
* **P4 GenerationBench** — vary the *methodology*, hold the model fixed at the local 7B.

Together they license: *surface correctness is what capability buys; nothing downstream of execution
moved.* They do **not** license any statement about the interaction — whether reflection, GoT or MCTS
help a frontier model as much as they help a 7B is untested by both, and the obvious rival
explanation for any P4 result is that scaffolding substitutes for capability and would do nothing on
a strong model. **That sentence belongs in Threats to Validity, and the third experiment that would
settle it is not being run** (RULE 5, no paid API, PI ruling 2026-08-07).
