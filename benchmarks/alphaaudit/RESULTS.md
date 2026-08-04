# AlphaAudit — RESULTS

**Benchmark version 1.2**, 2026-08-02. Version 1.0 contained a sign error in the abstention
ranking that inverted the semantic layer's contribution; the four semantic-inclusive configurations
in §9 were rerun with the PI's explicit authorisation and are the numbers shown here. Everything
else — corpus, audit labels, backtests, holdout contents, and the three non-semantic
configurations, which are bit-identical — is unchanged from v1.0. Version 1.2 changed no number at
all: it discloses that **27 of the 174 survivors are not deterministic functions of their inputs**,
so their recorded Sharpe is one draw rather than a fixed quantity (§11.1). Full detail for both in
[`CORRECTIONS.md`](CORRECTIONS.md). **The conclusions are unchanged: the null is not rejected.**

**Reference implementation, frozen 2026-08-01.** Every number below is transcribed from a
committed artifact; none was regenerated for this document. Where a number is unavailable it
says so and gives the reason, rather than being omitted.

**Read this first:** all performance figures are **net of Indian transaction costs** at a
₹10,00,000 book, retail depository mode. An earlier gross-of-costs evaluation of the same corpus
exists under `runs/superseded_gross/` and is retired; it may not be quoted without the word
"gross".

---

## 0. Provenance

| Field | Value |
|---|---|
| Generator | `qwen2.5:7b-instruct-q4_K_M` (id `845dbda0ea48`) via Ollama, local |
| Prompt digest (SHA256) | `f307433c7bda8595d52432b3bcb4f723663bfe706112a41f84e4beacfbde9934` |
| Temperature | 0.8 |
| Seed rule | global 42; candidate `i` drawn at `42 + i` |
| Semantic auditor | same model, same tag |
| Development window | 2020-01-01 → 2024-12-31, **1,232 sessions** |
| Holdout window | 2025-01-01 → 2025-12-31, **246 sessions** |
| Universe | top 100 NSE equities by trailing 126-session median traded value, quarterly, buffered |
| Price source | NSE bhavcopy archives (authoritative); `yfinance` used only as a cross-check |
| Book size | ₹10,00,000 |
| Depository mode | retail, ₹15.34 per scrip per sell instruction |

**The universe is a liquidity proxy and is not the NIFTY 100.** Free point-in-time index
membership could not be sourced. It is survivorship-free by construction, which was the binding
requirement.

---

## 1. Data foundation

| Quantity | Value | Sample |
|---|---|---|
| Trading sessions in calendar | 1,475 | 2019-01-01 → 2024-12-31 |
| Bhavcopy sessions cached | 1,359 / 1,359 | zero gaps |
| Panel rows | 2,287,802 | 2,697 symbols, EQ series |
| Quarterly rebalances | 20 | 2020-01 → 2024-10 |
| Universe size at each rebalance | exactly 100 | 20 / 20 |
| Distinct names ever in universe | **185** | vs 100 at any one time |
| Names added per rebalance | min 2, median 6, max 10 | 19 transitions |
| **Names in first universe, gone by last** | **41** | the survivorship evidence |
| Holdout universe carry-over | 87 of 100, 13 new entrants | 4 rebalances, 120 distinct |

**Corporate actions.** 52 declared for universe names; 43 applied (corroborated by prices);
5 disputed and deliberately **not** applied; 4 unreconcilable (no adjacent price).
RELIANCE 1:1 bonus: largest one-day move 49.8% raw → **0.9% adjusted**.

**Cross-check against `yfinance`** — reported, not reconciled:

| Comparison | Disagreement rate | Sample |
|---|---|---|
| Raw closes | 23.03% | 36,990 points, 30 symbols |
| Adjusted closes | **17.47%** | 35,757 points, 30 symbols |

Worst offenders are *exact* integer ratios (BAJFINANCE 10.000×, ASHOKLEY 2.000×, HDFCBANK
2.000×) on securities with no declared corporate action in the window. Genuine data noise is
never exactly 10×. The exchange file is designated authoritative. Residual RELIANCE 9.4% and
ITC 3.8% discrepancies are ours, arising from distributions we do not adjust for.

---

## 2. Engine

| Quantity | Value | Sample |
|---|---|---|
| Buy-and-hold (equal-weight universe), annualised return | 19.98% | 1,232 sessions |
| Annualised volatility | 21.14% | 1,232 |
| Sharpe | 0.94 | 1,232 |
| Max drawdown | −40.05% | 1,232 |
| NIFTY 100 **price** index, annualised return | 15.04% | 1,232 |
| **Tracking error, annualised** | **7.64%** | 1,232 aligned |
| **Correlation with index** | **0.9334** | 1,232 |

Benchmarked against the price index, not total return: the panel is a price-return series and a
TR benchmark would inflate tracking error artificially.

**Tracking-error decomposition: FAILED, reported as such.** Total 0.0775 over 1,231 sessions;
the two attempted legs came out 0.1152 and 0.1463, each exceeding the total they were meant to
partition. The traded-value proxy does not stand in for capitalisation weighting. Only the total
is reported. Separating the legs needs point-in-time free-float market capitalisation, which
could not be obtained free.

---

## 3. Fixtures

### Leaky (positive controls) — gross of costs

| Fixture | Sharpe | Sharpe, artifacts removed | Ann. return | Max DD | Sessions |
|---|---|---|---|---|---|
| `leak_future_return` | **12,098** | 11,389 | 356,661% | −1.4% | 1,232 |
| `leak_full_sample_scaler` | **2.33** | 2.28 | 60.2% | −31.2% | 1,232 |
| `leak_survivorship` | **1.23** | 1.25 | 25.8% | −38.7% | 1,232 |

Each holds its number when registered data artifacts are removed, confirming it cheats for its
intended reason rather than riding a data scar.

### Clean

30 honest strategies (moving-average crossover, momentum, mean reversion, volatility-scaled,
etc.), forming the false-positive control.

---

## 4. Transaction cost model

**Round trip, ₹1,00,000 delivery trade** — verified by the PI against a discount broker's public
calculator, and pinned as a unit test asserted to the paisa:

| Component | Amount (₹) |
|---|---|
| STT, both legs | 200.00 |
| Exchange transaction charge + IPFT, both legs | 6.14 |
| SEBI turnover fee | 0.20 |
| Stamp duty (buy leg only) | 15.00 |
| GST at 18% on charges (not on taxes) | 1.14 |
| **Total, broker-comparable** | **222.4812** |
| Depository charge, retail (sell leg, per scrip) | 15.34 |
| **Total, as charged in this study** | **237.8212** |

Rates are a schedule of three complete epoch snapshots (2015-01-01 baseline, 2024-04-01 NSE 1%
reduction, 2024-10-01 SEBI true-to-label), selected by trade date. A trade before the schedule
begins raises rather than silently using the nearest rate.

**Slippage (linear in volume participation) and impact (square-root law) coefficients are
engineering defaults, not calibrations to Indian data.** Daily bars cannot support a calibration
and none is claimed.

**Unsourced rates, held constant, each documented with its maximum error:** pre-2023 IPFT,
pre-2021 SEBI turnover fee, the 2021–23 NSE hike-and-rollback. Jointly under one part in a
thousand of a round trip. Stamp duty before 2020-07-01 was state-dependent; the uniform 0.015%
overstates cost in the first six months of the window.

---

## 5. The corpus funnel

**n = 1,550 draws**, two batches, pooled only after asserting identical model tag, prompt digest,
seed rule and window.

| Stage | Count | Share |
|---|---|---|
| Draws requested | 1,550 | — |
| Returned parseable, interface-conforming code | **1,550** | 100.0% of draws |
| Executed to completion against real data | **631** | 40.7% of corpus |
| — flat: ran 1,232 sessions, never traded | **406** | 64.3% of executed |
| — **rankable: executed and traded** | **225** | **14.5% of corpus** |
| Runtime error | 894 | 57.7% |
| Timeout at 180 s | 25 | 1.6% |

### Runtime error taxonomy, n = 894

| Exception | Count | Share of errors |
|---|---|---|
| `AttributeError` | 327 | 36.6% |
| `ColumnNotFoundError` | 247 | 27.6% |
| `TypeError` | 159 | 17.8% |
| `ZeroDivisionError` | 40 | 4.5% |
| `ValueError` | 29 | 3.2% |
| `KeyError` | 25 | 2.8% |
| `InvalidOperationError` | 21 | 2.3% |
| `NameError` | 17 | 1.9% |
| `IndexError` | 14 | 1.6% |
| `SchemaError` | 3 | 0.3% |

`AttributeError` + `ColumnNotFoundError` = **64.2% of all failures**, and both trace
overwhelmingly to pandas methods called on polars frames, or a wide frame treated as long. The
prompt states the distinction explicitly and carries a worked example.

**The auditor was never reached for 57.7% of the corpus, and had nothing to deflate for a
further 26.2%.**

---

## 6. Gross → net, development window, n = 225 traded

Both figures come from a **single run**: the engine records the pre-cost series alongside the
post-cost one.

| Quantity | Value | Sample |
|---|---|---|
| Best Sharpe, gross | 1.3049 | 225 |
| **Best Sharpe, net** | **1.2807** | 225 |
| Mean Sharpe reduction | 0.5867 | 225 |
| **Median Sharpe reduction** | **0.1943** | 225 |
| Mean CAGR reduction | 0.5813 | 225 |
| Mean turnover, annualised | 82.2× | 225 |
| Median turnover, annualised | 17.6× | 225 |
| Mean cost drag, annualised | 58.13% | 225 |
| Median cost drag, annualised | 4.39% | 225 |
| **Sharpe sign reversed, +ve → −ve** | **35 (15.6%)** | 225 |
| **Sharpe sign reversed, −ve → +ve** | **0** | 225 |
| Ruined outright (equity reached zero) | 11 | 631 |

Mean CAGR reduction equals mean cost drag exactly (0.5813) — an identity that has to hold, and a
wiring check. The −ve → +ve count must be 0; costs cannot improve a result.

### Top of the corpus by gross Sharpe

| candidate | gross | net | drag | turnover/yr | cost/yr |
|---|---|---|---|---|---|
| candidate_478 | 1.3049 | **1.2807** | −0.024 | 2.8× | 0.4% |
| candidate_298 | 1.2307 | 1.1023 | −0.128 | 24.0× | 3.4% |
| candidate_081 | 1.2245 | 0.8495 | −0.375 | 40.5× | 5.5% |
| candidate_141 | 1.0211 | 0.2524 | −0.769 | 11.2× | 2.6% |
| candidate_1207 | 1.0112 | 0.1304 | −0.881 | 16.4× | 15.1% |

**The best gross strategy is also the best net strategy, because it barely trades.** Cost
survival selects for low turnover, not for skill.

---

## 7. Positive control — eleven standard factors

| strategy | gross | net | drag | turnover/session | cost bps/session |
|---|---|---|---|---|---|
| equal_weight_universe | 0.9728 | 0.9615 | −0.011 | 0.003 | 0.10 |
| long_term_reversal_756d | 0.7443 | 0.6558 | −0.089 | 0.036 | 0.48 |
| high_volatility | 0.6134 | 0.4864 | −0.127 | 0.076 | 1.77 |
| momentum_skip_month | 1.1261 | **0.9656** | −0.161 | 0.120 | 1.69 |
| low_volatility | 1.0231 | 0.6093 | −0.414 | 0.125 | 2.37 |
| relative_strength | 0.9449 | 0.3038 | −0.641 | 0.159 | 5.40 |
| dual_momentum_21_126 | 0.8267 | **−0.2792** | −1.106 | 0.329 | 8.66 |
| inverse_volatility_weighted | 0.8239 | **−1.3792** | −2.203 | **0.011** | **10.18** |
| mean_reversion_5d | −0.2836 | −1.5286 | −1.245 | 0.869 | 13.68 |
| bollinger_reversion | −1.3722 | −3.0937 | −1.721 | 0.907 | 18.28 |
| random_walk_baseline | −1.2261 | −3.1134 | −1.887 | 1.796 | 108.32 |

Drag rises with turnover in **10 of 11** rows. **Two factors flip from profitable to
unprofitable.** Momentum survives at 0.9656 but no longer beats equal-weighting (0.9615) — net
of Indian costs, the momentum premium on this universe is inside the noise.

### The row that breaks monotonicity: book-size sensitivity

DP contribution measured by **differencing against a no-DP ablation**, not recomputed from the
rate:

| strategy | book | total bps/day | DP bps/day | DP share | Sharpe |
|---|---|---|---|---|---|
| inverse_volatility_weighted | ₹10 L | 10.183 | 9.963 | **97.8%** | **−1.3792** |
| inverse_volatility_weighted | ₹1 Cr | 0.866 | 0.597 | 69.0% | **+0.6359** |
| inverse_volatility_weighted | ₹10 Cr | 0.590 | 0.059 | 10.0% | **+0.6922** |
| momentum_skip_month | ₹10 L | 1.685 | 0.066 | 3.9% | 0.9656 |
| momentum_skip_month | ₹1 Cr | 2.103 | 0.007 | 0.3% | 0.9258 |
| momentum_skip_month | ₹10 Cr | 3.944 | 0.001 | 0.0% | 0.7506 |
| equal_weight_universe | ₹10 L | 0.096 | 0.010 | 10.4% | 0.9615 |
| equal_weight_universe | ₹10 Cr | 0.107 | 0.000 | 0.1% | 0.9602 |

DP falls **169× across a 100× book sweep**, close to inverse proportionality. The verdict on
`inverse_volatility_weighted` **reverses with assumed capital**. Momentum runs the opposite way —
cost rises with book size as impact grows while the fixed fee shrinks. **The fixed-fee and
proportional-cost regimes cross over.**

**₹10 lakh was kept as the reported default because it was the pre-existing configuration.** It
was not chosen after observing that it flips a sign.

### Depository regime sensitivity at ₹10 lakh (Sharpe)

| strategy | gross | retail ₹15.34 | research ₹3.50 | no DP |
|---|---|---|---|---|
| momentum_skip_month | 1.1261 | 0.9656 | 0.9704 | 0.9718 |
| inverse_volatility_weighted | 0.8239 | −1.3792 | 0.4663 | 0.7764 |
| dual_momentum_21_126 | 0.8267 | −0.2792 | 0.1268 | 0.2215 |
| relative_strength | 0.9449 | 0.3038 | 0.5799 | 0.6485 |

**Both flipped factors un-flip under research mode.** Retail is the headline; research is the
sensitivity.

### Cost composition

| strategy | statutory | DP | slippage | impact |
|---|---|---|---|---|
| momentum_skip_month | **79.6%** | 3.9% | 5.3% | 11.2% |
| equal_weight_universe | 32.7% | 10.4% | 54.9% | 1.9% |
| random_walk_baseline | 18.5% | **73.3%** | 7.2% | 1.0% |

For a realistically-trading strategy, **four fifths of the drag is published statutory rate**;
only ~16% comes from the two chosen coefficients.

---

## 8. Auditor layers

### A_static — fixtures

| Quantity | Value | Sample |
|---|---|---|
| Leaky fixtures caught | **3 / 3** | 3 |
| Honest fixtures wrongly flagged | **0 / 30** | 30 |
| Recall | 1.00 | 3 |
| Precision | 1.00 | 3 |

**Discount this.** With syntax-based detectors alone, recall was **1/3**. The decisive
structural rules were added after seeing that failure and were then measured on the same three
fixtures that motivated them — closer to training accuracy than to a test.

### A_static — corpus, n = 1,550 sources (needs no execution)

| Leak class | Count |
|---|---|
| `snooped_parameter` | 166 |
| `future_indexing` | 58 |
| `target_in_features` | 23 |
| `boundary_crossing_window` | 2 |
| `survivorship_selection` | 1 |
| `full_sample_statistic` | **0** |

222 of 1,550 rejected (14.3%); 195 carried one class, 26 two, 1 three. Of the 225 rankable
candidates, **28 (12.4%)** were static-rejected.

> **Corrected 2026-08-04.** This sentence previously read "26 (11.6%)", which contradicted the
> fate table immediately below it — that table has always said 28, and it is right. The error was
> confined to this file; the paper never stated the figure. See `CORRECTIONS.md`.

### Where the leak flags land — added 2026-08-01

| Fate of the 222 static-rejected candidates | Count | Share |
|---|---|---|
| Runtime error — never ran | 114 | 51.4% |
| Evaluated but flat — ran, never traded | 79 | 35.6% |
| Timeout | 1 | 0.5% |
| **Evaluated and traded** | **28** | **12.6%** |

**Only 28 leak findings in the corpus attach to a strategy that took a position.**

Worked example, `candidate_053` — flagged `future_indexing` at confidence 1.0 for a `shift(-1)`
filtered onto a past session date. The verdict is correct. The block is also **unreachable**: the
preceding guard requires 21 rows from `view.closes(lookback=20)`, which returns at most 20, so it
is false on every session. Had it executed it would have raised twice — invalid `date(days=1)`
construction, and `adj_close` absent from the pivoted wide frame.

**A leakage prevalence rate over generated source is not a deployment risk rate.** "14.3% of
candidates leak" and "28 of 1,550 leak *and* trade" are both true; only the second is
decision-relevant.

*Provenance: this measurement did not exist until the PI's Checkpoint 1.5 Q5 answer prompted the
question. It is not a repackaging of an earlier number.*

**The zero is not credible and is reported as an instrument blind spot.** The prompt advises
plain Python over polars expression chains to raise the executable rate, and expression chains
are where this class arises. The distribution describes *this prompt*, not LLM-generated
strategies in general.

### A_stat — deflation

| Quantity | Value | Sample |
|---|---|---|
| **Trial count N** | **1,887** | ledger 1,877 + 10 documented unrecorded retries |
| Ledger entries, hash chain verified | 1,877 | append-only JSONL |
| Rejected (DSR probability < 0.95) | **631 / 631 (100%)** | 631 |
| **Highest DSR probability in corpus** | **4.37 × 10⁻⁸** | candidate_219 |
| Best raw Sharpe → its DSR probability | 1.3049 → 5.9 × 10⁻⁵ | 1 |
| **PBO, pooled corpus** | **unavailable** | see below |
| PBO, 12-fixture suite | 0.588 (0.531–0.667 across split counts) | 1,232 × 12 |

**The +10 correction is applied at the point of use, not written into the ledger.** An
append-only ledger edited when found wrong is not tamper-evident. The file says 1,877 and
deflation uses 1,887; the discrepancy is deliberate.

**PBO is unavailable on the pooled corpus** because the 11 ruined strategies produce short series
and the common-length matrix loses too many columns. It was **not** repaired by dropping them —
that restriction was not part of the intended protocol.

**RESULT, not caveat — the rejection rate is uninterpretable at this trial count.** At N=1,887
over 1,232 sessions every evaluated strategy is rejected. **Whether this reflects appropriate
statistical correction or saturation of the test cannot be determined from these data.** The
deflation term grows with N while sample length is fixed, so at some ratio the test rejects
everything regardless of merit.

This bears on §9, and more sharply than first reported. The layer rejects all 631 it scores at a
recorded confidence of exactly 1.0, so it carries **no ordering information at all** among scored
candidates — every one ties, and its only discrimination is between scored and unscored. The four
configurations containing it are the four with the steepest fall at tight coverage. The rejection
rate and the ranking failure are plausibly the same fact observed twice.

**Recorded as an observed property of this implementation, not repaired.** Per the PI's ruling of
2026-08-01, the layer is not modified in this project and no further holdout evaluation is
authorised for it. A genuinely continuous statistical confidence measure is future work. Separating them requires evaluating the layer across
trial-count regimes, which this corpus cannot support because N is a property of the experiment
rather than a free parameter.

### A_sem — semantic

| Quantity | Value | Sample |
|---|---|---|
| **Cohen's κ** | **0.5890** | 50 hand-labelled items |
| Raw agreement | 0.7600 (38/50) | 50 |
| Landis & Koch band | moderate (0.41–0.60), at the top | — |
| Items classified, failures | 50 / 50, zero failures | 50 |
| Wall clock | 3.0 min against a 5.9 min projection | 50 |

Per-class agreement:

| Class | Reviewer | Model | Agreed |
|---|---|---|---|
| `consistent` | 31 | 29 | 26 |
| `rationale_implementation_mismatch` | 9 | 9 | 5 |
| `unacknowledged_known_anomaly` | 5 | 8 | 5 |
| `unfalsifiable_mechanism` | 5 | 4 | **2** |

**No prompt was tuned before this measurement and none after it.** The decision to keep the layer
was taken before the number was known.

**κ limitations that must accompany any quotation:** labels are reviewer-confirmed over a
model-drafted first pass, not from scratch; class balance is engineered (17 of 50 constructed);
17 fixtures appear twice; two leaky fixtures retain `# THE CHEAT:` comments.

### A_sem — corpus, n = 1,550

| Label | Count | Share |
|---|---|---|
| `consistent` | 1,396 | 90.1% |
| `rationale_implementation_mismatch` | 119 | 7.7% |
| `unacknowledged_known_anomaly` | 21 | 1.4% |
| `unfalsifiable_mechanism` | 14 | 0.9% |

Rejected: **154 / 1,550 (9.9%)**. Whether the 154 are the *right* ones is not established on this
population — κ was measured on a different one.

### Layer independence

`leak_full_sample_scaler` is caught by `A_static` and **correctly** passed by `A_sem`: its
rationale honestly describes what its code does, and its only defect is where the scaler was
fitted. **A leaky strategy can carry an honest rationale.** The layers are not redundant, and any
evaluation collapsing them into one "flagged by something" signal would obscure that.

---

## 9. The abstention frontier — THE HEADLINE

**Population n = 225**, fixed by *development* eligibility, scored on the holdout.

| Quantity | Value |
|---|---|
| Mean development Sharpe (net) | −0.1896 |
| **Mean holdout Sharpe (net)** | **−1.0351** |
| Best holdout Sharpe | +1.3618 |
| **Random-rejection AUAP 95% interval** | **[−1.2208, −0.8600]** |

### Holdout ablation, all seven configurations

| Layers | AUAP | P(0.05) | P(1.0) | Beats random |
|---|---|---|---|---|
| semantic | −1.1967 | −1.1347 | −1.0351 | no |
| static | −1.2529 | −1.1176 | −1.0351 | no — **below the interval** |
| static + semantic | −1.2782 | −1.1805 | −1.0351 | no — **below the interval** |
| statistical | −1.3497 | −3.0643 | −1.0351 | no — **below the interval** |
| semantic + statistical | −1.3777 | −3.2931 | −1.0351 | no — **below the interval** |
| static + semantic + statistical | −1.4184 | −3.2931 | −1.0351 | no — **below the interval** |
| static + statistical | −1.4339 | −3.2931 | −1.0351 | no — **below the interval** |

**THE NULL IS NOT REJECTED.** One configuration sits inside the random interval; six sit below
it, ordering strategies *worse* than chance.

Most curves **fall** as coverage tightens. Concentrating on the candidates the auditor trusts
most selects a worse portfolio than taking everything.

### Development ablation — DIAGNOSTIC ONLY

Random band [−0.3706, −0.0270]; population mean −0.1896.

| Layers | AUAP | P(0.05) | Beats random |
|---|---|---|---|
| semantic | −0.2917 | +0.2910 | no |
| static + semantic | −0.3310 | +0.3142 | no |
| static | −0.3327 | +0.2073 | no |
| statistical | −0.3995 | −1.3599 | no |
| semantic + statistical | −0.4124 | −1.3494 | no |
| static + semantic + statistical | −0.4374 | −1.3494 | no |
| static + statistical | −0.4550 | −1.3494 | no |

A correctly-oriented semantic layer does concentrate the better strategies at tight coverage
**in sample** — `P(0.05)` of +0.2910 alone and +0.3142 with the static layer, against a population
mean of −0.1896. None of that selectivity survives to the holdout, where the same two rankings give
−1.1347 and −1.1805. That gap is this paper's subject, and under v1.0's inverted ranking it was
invisible: see [`CORRECTIONS.md`](CORRECTIONS.md).

### The circularity diagnosis

On the **gross** development evaluation the `statistical` layer alone was the only configuration
to beat random. This was predicted in advance to be an artifact: its confidence is `1 − DSR`, and
DSR is a monotone function of the very development Sharpe that `P(c)` measures. Ranking by a
quantity and scoring on that same quantity beats random by construction.

Out of sample the comparison is legitimate, and the layer lands **fourth of seven, below the random
interval**, with the worst `P(0.05)` of any single layer (−3.0643). The diagnosis was made by
argument beforehand and confirmed by experiment afterwards. (Under v1.0 it ranked second-worst;
the change is an artifact of the four semantic-inclusive configurations moving down past it, not of
any change to this layer, whose numbers are bit-identical across versions.)

### The in-sample to out-of-sample collapse

| Quantity | Development | Holdout |
|---|---|---|
| Fraction positive (gross corpus, n = 221) | 216 / 221 | 105 / 221 |
| Correlation between the two | | **+0.4062** |

**The best out-of-sample performer had a development Sharpe of −1.6696.** `candidate_250`:
development **−1.6696**, holdout **+1.3618**. A strategy poor in sample and strong out of sample
is a coin landing well. This is the multiple-testing phenomenon appearing inside our own results.

---

## 10. Holdout log — CLOSED

| # | Date | Window | Pipeline | Authorisation | Status |
|---|---|---|---|---|---|
| 1 | 2026-07-31 | 2025 | gross of costs (no cost model existed) | PI, Checkpoint 1.4 Q1 | **retired**, frozen at `runs/superseded_gross/` |
| 2 | 2026-07-31 | 2025 | net of costs, corrected | PI, Checkpoint 1.1 Q8 | **superseded**, retained at `runs/pooled/ablation_holdout_v1.0_superseded.json` |
| 3 | 2026-08-01 | 2025 | net of costs, corrected semantic ranking | PI, 2026-08-01, quoted in `CORRECTIONS.md` | **final** |

**The holdout is permanently closed.**

Evaluation 3 was authorised in advance to repair a verified implementation bug (`CORRECTIONS.md`),
not in response to a disappointing number. **The previous version of this section stated that no
third evaluation was available under any circumstance; that sentence was overtaken by the PI's
ruling and is replaced here rather than quietly dropped.** No parameter, threshold, prompt or model
was changed after any of the three.

The holdout window was frozen in `config.yaml` **before any holdout data was fetched**, and set
to a calendar boundary rather than to the latest available session, so that window length could
not become a researcher degree of freedom.

**The correction changed the magnitudes and left the conclusion identical:** holdout mean moved
−0.2755 → −1.0351, and no configuration beat random in either evaluation.

---

## 11. Survivors handed forward

**174 candidates** cleared `A_static ∧ A_semantic`, frozen at `benchmarks/alphaaudit/survivors/`
with all three layer verdicts recorded against each. Net Sharpe range **−10.657 to 1.281**.

Under **all three** layers the survivor set is **exactly 0**, because the statistical layer
rejects the entire corpus. Reproduce with `--layers static semantic statistical`.

**This is the one place where a definition was chosen because it produces a non-empty answer, and
it is flagged as such.**

### 11.1 Determinism of the survivor set — added 2026-08-02

Run one of these strategies twice on the same panel with the same code and 27 of them return a
different Sharpe. Measured by `scripts/calibrate_tier1.py`, n = 185 (174 survivors + 11 standard
factors). Unlike every other number in this document these are **regenerated, not transcribed from
a committed artifact**: the run writes `data/processed/tier1_calibration.json`, and `/data/` is
gitignored, so the command below is the record rather than the file.

| Population | Deterministic | Not | Sample size |
|---|---|---|---|
| Survivors | **147** | 27 | 174 |
| Standard factors | **11** | 0 | 11 |
| Total | **158** | 27 | 185 |

Two mechanisms, both confirmed in source. **Unseeded randomness** in 26 of the 27 — `candidate_1011`
returned −3.4386, −4.2835 and −1.6182 on three consecutive runs in one process. **Hash-order
dependence** in 27 of 27, e.g. `candidate_002` line 42, `list(set(volatility_factors))[:10]`, which
selects an arbitrary ten symbols; its Sharpe was 0.7652 / 0.4566 / 0.6825 under `PYTHONHASHSEED`
0 / 1 / 2. Largest swing between seeds: `candidate_1011`, |Δ| = **2.5352**.

`PYTHONHASHSEED` is set nowhere in this repository, so the corpus was backtested with hash
randomisation active. **This is a charter RULE 6 failure in the harness's coverage**, and it is
recorded as one.

**What it does not touch.** Survivor membership: the auditors read code, and the only performance
gate is `FLAT_TOLERANCE = 1e-9` against a smallest absolute Sharpe of 0.0111 among the 27. The
headline null: extra noise makes an ordering harder to detect, never easier. **What it does touch:**
the §9 AUAP figures are computed from these Sharpes and carry the noise, so their trailing digits
are not exact.

These 27 are excluded from Project 2, whose fragility metric is a variance and cannot separate a
strategy's own noise from its regime response. **That exclusion is not performance-neutral** —
excluded mean −0.6722, retained mean −0.1320, all 174 −0.2158 — and no figure computed on the 147
may be quoted without it. See [`CORRECTIONS.md`](CORRECTIONS.md) v1.2.

```bash
python scripts/calibrate_tier1.py --workers 24     # ~17 min, 740 backtests, prints the table above
```

---

## 12. Process failures, reported

1. **Six candidates silently reclassified as runtime errors** during the first net run. `sum()`
   over an empty book returns integer zero, so the turnover column was inferred `Int64` and raised
   on the first genuine float. **Exit code was 0.** Among the six was `candidate_250`, holder of
   the highest holdout Sharpe in the previous evaluation. Caught only by comparing funnel counts
   against the retired gross run — a check imposed by review, not by the code. Fixed by `float()`
   coercion; after the fix, batch 1 showed zero outcome changes versus gross.

2. **90 minutes of GPU output destroyed and redone.** Running the audit with `--skip-semantic`
   overwrote batch 2's semantic verdicts with an empty dict. Caught by inspecting output rather
   than the exit code, which was zero. Every semantic figure above is from the recomputed run.
   The script now carries forward existing verdicts.

3. **A plotting bug caught before the figure was published.** The first holdout figure drew the
   *development* random band beneath the holdout curves. Every number in it was correct; the
   picture would have been actively misleading. Fixed, both figures regenerated.

4. **Two reporting imprecisions corrected during write-up.** The corpus maximum DSR probability
   was reported as `0.0000` in `reports/checkpoint_1.1_net_rerun.md` §3i; the true value is
   `4.37 × 10⁻⁸`. And the claim that all seven holdout configurations sat *inside* the random
   band was wrong — six fall below it. That correction was itself misstated as "two" when first
   written up, and stood wrong through v1.0; see `CORRECTIONS.md`, "A second error".

5. **The harness never pinned `PYTHONHASHSEED`, and nothing checked that a backtest was
   repeatable.** Every layer had tests; no test ran the same strategy twice and compared. The gap
   went unnoticed through the whole of Phase 1 and was found only when Project 2 needed a variance
   to mean something. Charter RULE 6 forbids unseeded randomness, and the harness had no way to
   detect a violation of it — the check that would have caught this is one line, and it was the
   absence of that check, not any individual script, that failed. See §11.1.

---

## 13. Reproduction

```bash
bash scripts/run_net_pipeline.sh      # full chain; the holdout steps will not re-authorise
python scripts/gross_vs_net.py --results runs/pooled/backtest_results.json
python scripts/dp_sensitivity.py
python -m pytest -q                   # tests/costs/test_india.py pins the ₹222.4812 round trip
```

The holdout steps in that script carry a spent authorisation string and **must not be re-run**.
