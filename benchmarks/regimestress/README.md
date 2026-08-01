# RegimeStress — when does a strategy break?

**Status: released, not started.** No code, no data, no results. This file exists to hold the
project's ground truth — its input set and its known methodological fork — so that neither is
reconstructed from memory when work begins.

P1 asked *is this signal real?* P2 asks *under what conditions does it stop working?*

---

## The problem, stated honestly up front

A strategy that performed well over the development window was tested against **one** sequence of
history. Indian markets since 2015 contain approximately one pandemic crash, one full rate cycle,
and one structural liquidity shift. **The number of independent regime observations is very
small.** Any claim of the form "this strategy is robust across regimes" rests on a handful of
effective observations, and the project's central contribution is addressing that scarcity
principally rather than pretending it away.

---

## Input set — fixed at Checkpoint 1.5 by PI override

The original plan was to stress-test P1's survivors alone. **The PI rejected that**, and the
reasoning is recorded here because it constrains everything downstream:

> If every input strategy is already statistically indistinguishable from noise, then you are
> studying the robustness of noise.

All 174 P1 survivors fail the statistical layer at *N* = 1,887 — as does every other candidate in
the corpus. Stressing only that population would measure how noise behaves under regime shift.

**The input set is therefore two populations, kept distinguishable at every stage:**

| Population | Count | Source |
|---|---|---|
| P1 survivors — cleared static + semantic, all failing statistical | 174 | [`../alphaaudit/survivors/`](../alphaaudit/survivors/) |
| Standard academic factor panel | 11 | the P1 positive control — equal-weight, momentum, value, quality, minimum variance, low volatility, and the remainder |

The factor panel is what turns this into a RegimeStress *benchmark* rather than RegimeStress on
weak LLM strategies. Results must be reportable separately for each population; pooling them would
hide whichever effect is real.

---

## Open methodological fork — must reach the PI before any code

**Regime labels are fit on data.** Fitting an HMM on the full sample and then testing strategies
within the resulting regimes is leakage — structurally the same failure P1 was built to detect.
The choice is between expanding-window regime fitting and full-sample labels with the limitation
stated prominently, and it is a PI decision, not an implementation detail.

Phase 2.0 halts on this. Nothing gets written until it is answered.

---

## What will live here

Nothing does yet. Listed so the shape is visible.

| Path | What it will hold |
|---|---|
| `regimes/` | the fitted regime labels and their state-occupancy diagnostics |
| `events.md` | hand-curated, dated, sourced SEBI structural-event timeline |
| `RESULTS.md` | every number with its sample size, as in P1 |
| `fragility/` | fragility scores per strategy across real and synthetic regimes |

---

## Inherited from P1, unchanged

The cost model, the point-in-time engine, the survivorship-free universe, and the seeding and
manifest discipline are shared infrastructure — see [`../../PROJECTS.md`](../../PROJECTS.md). P2
does not re-implement them, and any change to them alters P1's published numbers as well.

**The P1 holdout is permanently closed.** P2 requires its own held-out window, designated before
any P2 result is computed and never touched afterwards without written authorisation.
