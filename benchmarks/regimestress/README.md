# RegimeStress v1 — a benchmark for measuring when a strategy breaks

**Status: FROZEN, 2026-08-02.** Results in [`RESULTS.md`](RESULTS.md). Corrections in
[`CORRECTIONS.md`](CORRECTIONS.md). Paper:
[`../../papers/regimestress.tex`](../../papers/regimestress.tex).

P1 ([AlphaAudit](../alphaaudit/)) asked *is this signal real?* RegimeStress asks *under what
conditions does it stop working?*

---

## The problem, stated honestly up front

A strategy that performed well over the development window was tested against **one** sequence of
history. Indian markets since 2015 contain approximately one pandemic crash, one full rate cycle,
and one structural liquidity shift. **The number of independent regime observations is very
small.** Any claim of the form "this strategy is robust across regimes" rests on a handful of
effective observations, and the benchmark's central contribution is addressing that scarcity
principally rather than pretending it away.

## What the benchmark provides

| Component | What it is |
|---|---|
| **Causal regime labels** | Expanding-window HMM with a forward-only decode. No label uses information from after the session it describes. |
| **Counterfactual Regime Resampling** | Stationary block bootstrap conditioned on regime label. Resamples *dates*, so cross-sectional correlation survives intact. |
| **A moment-validation suite** | Nine statistics the synthetic paths must reproduce before any strategy runs on them. |
| **Two stress tiers** | One faithful and expensive, one cheap — with a measured account of when each is adequate. |
| **A frozen population** | With exclusion criteria fixed *before* any stress path was run. |
| **A fragility-prediction task** | The benchmark's first task, on which the reference implementation returns a diagnosed null. |
| **A generated results pipeline** | Every number in `RESULTS.md` and in the paper is emitted from the artifacts by one script. |

## The three results worth knowing before you use it

1. **The cheap tier does not substitute for the expensive one.** Tier 2 agrees with Tier 1 at
   Spearman 0.897 on mean performance and 0.620 on fragility. Resampling realised returns preserves
   how a strategy *did* and loses most of how much it *moves*. Do not use Tier 2 as a fragility
   proxy — we intended to, and measured that we should not.

2. **One shortcut does hold.** Across-regime fragility computed directly from a persisted return
   series agrees with its bootstrapped counterpart at Spearman 0.963. The expensive experiment was
   run in full and then used to validate the free computation, rather than skipped and justified
   afterwards.

3. **Fragility prediction is a null at this sample size.** Strategy characteristics carry
   statistically detectable information about the fragility *ranking* and essentially none about
   its *level*. The limitation is data variance, not model capacity, and it is diagnosed rather
   than asserted. See [`RESULTS.md`](RESULTS.md) §7.

## What is deliberately *not* here

* No tuned model. The diagnosis showed the constraint was variance, so no capacity was added.
* No claim that the synthetic histories are the true counterfactual distribution. See the paper's
  Threats to Validity, which is a section and not a footnote.
* No holdout evaluation. P1's holdout is permanently closed; RegimeStress does not touch it.

---

## Contents

| Path | What it holds |
|---|---|
| `RESULTS.md` | **Generated.** Every measured result. Do not edit — edit `RESULTS.template.md`. |
| `RESULTS.template.md` | The source of `RESULTS.md`, with double-brace placeholders. |
| `paper_numbers.json` | Every published figure, with the artifact it came from. The audit trail. |
| `CORRECTIONS.md` | Five defects found in this benchmark, including two of our own that had inflated a published result. |
| `sebi_events.csv` | 18 dated, sourced market-structure events, each carrying an evidence grade. |
| `excluded_nondeterministic.json` | The 27 strategies that are not deterministic functions of their inputs. Frozen before any stress path ran. |
| `knife_edge.json` | The strategies whose Sharpe is unstable under a numerically negligible perturbation. |
| `knife_edge_stability.json` | What that exclusion costs, quantified. |
| `duplicates.json` | 11 clusters of strategies with identical realised return series. A property of the generator, shipped as data. |

Source lives in [`../../src/stress/`](../../src/stress/): `regimes.py` (causal labelling),
`crr.py` (the resampler), `moments.py` (validation), `panel.py` (synthetic panel construction),
`tier2.py`, `fragility.py`, `characteristics.py`, `predictor.py`, `diagnosis.py`, `narrative.py`,
`events.py`, `inputs.py`.

Generated into gitignored `data/processed/`: `regime_labels.parquet` (the only labels anything
downstream may use), `regime_diagnostics.json`, `regime_k_selection.json`, `crr_calibration.json`,
`tier1_calibration.json`, `tier2_fragility.json`, `tier_comparison.json`, `rerun_decision.json`,
`fragility.json`, `characteristics.parquet`, `fragility_predictor.json`,
`predictor_diagnosis.json`, `stress_narratives.json`, `factor_design.json`.

### The event timeline

18 events, **12 effective inside the development window**, spanning circuit breakers, physical
settlement of stock derivatives, the COVID volatility measures, margin pledge and peak margin, the
T+1 migration, exchange-charge and STT revisions, and the 2024 index-derivatives framework. Every
row carries an `evidence` grade — from `primary_circular_pdf` down to `secondary_sources_agree` —
so a reader can see which entries rest on a circular and which on reporting.

**Known gaps are recorded in the file's `.meta.json` rather than left implicit**: macro and fiscal
events (demonetisation, GST, LTCG reintroduction) are out of scope; the CNX-to-NIFTY renaming is
visible in the NSE archive but no announcement date was sourced, so it is omitted rather than dated
by inference; index reconstitution is recorded as a periodic rule, not as dated rows. **The
timeline is not claimed to be exhaustive.**

---

## Input set — fixed at Checkpoint 1.5 by PI override

The original plan was to stress-test P1's survivors alone. **The PI rejected that**, and the
reasoning is recorded here because it constrains everything downstream:

> If every input strategy is already statistically indistinguishable from noise, then you are
> studying the robustness of noise.

All 174 P1 survivors fail the statistical layer at *N* = 1,887 — as does every other candidate in
that corpus. Stressing only that population would measure how noise behaves under regime shift.

**The input set is therefore two populations, kept distinguishable at every stage:** the 174
audited survivors, and an 11-member panel of standard academic factor strategies acting as a
positive control. The factor panel is what makes this a RegimeStress *benchmark* rather than
RegimeStress on weak LLM strategies. Results are reportable separately for each population;
pooling them would hide whichever effect is real.

## The methodological fork, resolved

**Regime labels are fit on data**, and fitting an HMM on the full sample before testing strategies
inside the resulting regimes is leakage — structurally the same failure P1 was built to detect.

**PI ruling, 2026-08-01: expanding-window only. No full-sample fit under any code path.** The HMM
is re-estimated at each of P1's quarterly rebalance dates on sessions strictly before it, and K is
selected once on pre-2020 data and frozen. Multi-start EM was added after a single-start fit
converged to a degenerate optimum; the PI approved this as a convergence fix, and it did not change
the selected K.

**The decode is the design.** Fitting on past data only is necessary but not sufficient:
`hmmlearn`'s `predict` (Viterbi) and `predict_proba` (smoothed) both decide the state at session
`t` using observations after `t`. Both are forbidden here. Labels come from a forward-only
recursion in [`../../src/stress/regimes.py`](../../src/stress/regimes.py) so the causal claim is
auditable in one place. The cost is a two-session lag at turning points — which is what an observer
standing at `t` could actually have known.

### Labels drift, and results must be reported accordingly

Terminal disagreement in 2022 is **62.3%**, against 9.4% in 2023 and 1.6% in 2024 — and every one
of the 154 relabels is *upward* by roughly one state. Calm 2023–24 entered the expanding window and
recalibrated what "normal" volatility means, so 2022 reads as elevated in hindsight.

**Consequence, binding on every downstream result:** per-year (or per-refit-period) regime
summaries are the **primary** analysis. Pooled summaries are supplementary and carry an explicit
note that regime definitions evolve under expanding-window estimation. "State 2 in 2022" and "state
2 in 2024" are not the same conditioning event. This is a limitation of causal refitting, not a
flaw.

---

## Extending the benchmark

RegimeStress is infrastructure to build on, not a solved problem. The open directions, in rough
order of how much they would add:

* **A larger and more diverse strategy corpus.** The one intervention our own diagnosis supports —
  the learning curve has not saturated at 109 rows. A corpus from several generators would also
  separate "fragility is hard to predict" from "fragility is hard to predict *for strategies this
  generator writes*".
* **Other markets and asset classes.** The apparatus is not India-specific.
* **Richer regime models.** Ours labels on volatility features alone. Liquidity, macro state and the
  market-structure timeline all ship here and are unused by the labeller.
* **Alternative stress-generation methods.** Block bootstrap is one choice; the moment suite is a
  ready yardstick for comparing others against it.
* **Intraday strategies.** Daily bars bound what can be said about the fast strategies our
  knife-edge filter removes.
* **Scoring future LLM strategy generators**, including on the duplicate structure of their output.

## Reproducing everything

```bash
python scripts/fetch_regime_burnin.py        # index history for HMM burn-in
python scripts/cross_check_burnin.py         # against NSE's published index bhavcopy
python scripts/select_regime_states.py       # BIC over K; write-once
python scripts/build_regimes.py              # causal labels + stability diagnostics
python scripts/build_sebi_events.py          # market-structure timeline
python scripts/calibrate_crr.py              # block length + moment validation
python scripts/calibrate_tier1.py            # determinism census + timing
python scripts/run_stress_tier1.py           # the expensive experiment (~124 CPU-hours)
python scripts/run_stress_tier2.py           # the cheap one (~23 seconds)
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
python scripts/build_paper_numbers.py        # regenerates RESULTS.md and the paper's macros
python scripts/check_paper_numbers.py        # fails if the paper states an ungenerated figure
```

Only `run_stress_tier1.py` is expensive. Everything else completes in minutes.

---

## Inherited from P1, unchanged

The cost model, the point-in-time engine, the survivorship-free universe, and the seeding and
manifest discipline are shared infrastructure — see [`../../PROJECTS.md`](../../PROJECTS.md).
RegimeStress does not re-implement them, and any change to them alters P1's published numbers as
well.

**The P1 holdout is permanently closed** and was not touched by any work in this benchmark.
