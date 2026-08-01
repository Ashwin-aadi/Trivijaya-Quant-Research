# RegimeStress — when does a strategy break?

**Status: Phase 2.0 complete and PI-approved (2026-08-01). Phase 2.1 not started.**

P1 asked *is this signal real?* P2 asks *under what conditions does it stop working?*

## Phase 2.0 — regime labelling, as built

Every development session 2020-01-01 to 2024-12-31 carries a market-regime label produced without
any information from after that session.

| | |
|---:|---|
| **4** | states, selected by BIC on 1,207 pre-2020 sessions, then frozen permanently |
| **1,233** | sessions labelled, across 20 quarterly refits |
| **filtered** | decode — `argmax_k P(state_t \| obs_1..t)`, never Viterbi, never smoothed |
| **2020-02-26** | first session labelled most-stressed; the COVID selloff began ~2020-02-24 |
| **9.5% / 21.8%** | one-quarter revision rate / terminal disagreement, pooled over 1,233 |
| **0 / 55** | burn-in sessions disagreeing with NSE's own published index close |

**The decode is the design.** Fitting on past data only is necessary but not sufficient:
`hmmlearn`'s `predict` (Viterbi) and `predict_proba` (smoothed) both decide the state at session
`t` using observations after `t`. Both are forbidden here. Labels come from a forward-only
recursion implemented in [`src/stress/regimes.py`](../../src/stress/regimes.py) so the causal claim
is auditable in one place. The cost is a two-session lag at turning points, which is what an
observer standing at `t` could actually have known.

### Labels drift, and results must be reported accordingly

Terminal disagreement in 2022 is **62.3%**, against 9.4% in 2023 and 1.6% in 2024 — and every one
of the 154 relabels is *upward* by roughly one state. Calm 2023-24 entered the expanding window and
recalibrated what "normal" volatility means, so 2022 reads as elevated in hindsight.

**Consequence, binding on Phase 2.2 by PI ruling:** per-year (or per-refit-period) regime summaries
are the **primary** analysis. Pooled summaries are supplementary and must carry an explicit note
that regime definitions evolve under expanding-window estimation. "State 2 in 2022" and "state 2 in
2024" are not the same conditioning event. This is a limitation of causal refitting, not a flaw.

### Reproducing

```bash
python scripts/fetch_regime_burnin.py      # 2015-2018 index history, fitting burn-in only
python scripts/cross_check_burnin.py       # vs NSE's published index bhavcopy
python scripts/select_regime_states.py     # BIC over K in {2,3,4}; write-once, exits 2 if rerun
python scripts/build_regimes.py            # labels + stability diagnostics
python scripts/build_sebi_events.py        # validate and publish the event timeline
```

---

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

## The methodological fork, resolved

**Regime labels are fit on data**, and fitting an HMM on the full sample before testing strategies
inside the resulting regimes is leakage — structurally the same failure P1 was built to detect.

**PI ruling, 2026-08-01: expanding-window only. No full-sample fit under any code path.** The HMM
is re-estimated at each of P1's quarterly rebalance dates on sessions strictly before it, and K is
selected once on pre-2020 data and frozen. Multi-start EM was added after a single-start fit
converged to a degenerate optimum; the PI approved this as a convergence fix, and it did not change
the selected K.

---

## Contents

| Path | What it holds |
|---|---|
| `sebi_events.csv` | 18 dated, sourced market-structure events — the canonical copy, version-controlled |
| `../../src/stress/regimes.py` | causal features, expanding-window fitting, filtered decode, canonicalisation |
| `../../src/stress/events.py` | timeline loader and its validation rules |
| `../../src/stress/inputs.py` | index series assembly (burn-in + calendar), overlap-checked |
| `RESULTS.md` | not written yet — due with Phase 2.3 |

Generated into gitignored `data/processed/`: `regime_labels.parquet` (the only labels anything
downstream may use), `regime_diagnostics.json` (stability, occupancy, timeline — deliberately
non-causal, never read as labels), `regime_k_selection.json` (frozen K), `regime_ledger.jsonl`.

### The event timeline

18 events, **12 effective inside the development window**, spanning circuit breakers, physical
settlement of stock derivatives, the COVID volatility measures, margin pledge and peak margin, the
T+1 migration, exchange-charge and STT revisions, and the 2024 index-derivatives framework. Every
row carries an `evidence` grade — from `primary_circular_pdf` down to `secondary_sources_agree` —
so a reader can see which entries rest on a circular and which on reporting.

**Known gaps are recorded in the file's `.meta.json` rather than left implicit**: macro and fiscal
events (demonetisation, GST, LTCG reintroduction) are out of scope; the CNX-to-NIFTY renaming is
visible in the NSE archive but no announcement date was sourced, so it is omitted rather than
dated by inference; index reconstitution is recorded as a periodic rule, not as dated rows. **The
timeline is not claimed to be exhaustive.**

---

## Inherited from P1, unchanged

The cost model, the point-in-time engine, the survivorship-free universe, and the seeding and
manifest discipline are shared infrastructure — see [`../../PROJECTS.md`](../../PROJECTS.md). P2
does not re-implement them, and any change to them alters P1's published numbers as well.

**The P1 holdout is permanently closed.** P2 requires its own held-out window, designated before
any P2 result is computed and never touched afterwards without written authorisation.
