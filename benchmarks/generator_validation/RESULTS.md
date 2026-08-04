# Generator validation — results

Every number here is written by `scripts/build_generator_validation_results.py`
from the run artifacts. None is transcribed, so a figure in the text cannot drift
from the run that produced it.

**Nothing is withheld.** Failed hypotheses, arms outside their pre-registered band,
and auditor layers that flagged nothing appear with the same prominence as the
results that worked.

## The subjects

| Arm | Model, as the interface reported it | Requests | Strategies |
|---|---|---|---|
| `gpt` | GPT, base model | 4 | 20 |
| `claude` | Claude Opus, high effort | 4 | 20 |
| `gemini` | Gemini Pro | 4 | 20 |

Each arm is four independent chat requests of five strategies, issued from the
frozen P1 prompt with no project context. Draws within one request are **not**
independent: the model wrote five in one pass with the earlier ones in its context.
Any interval that treats an arm as 20 free draws is therefore optimistic. That is
stated rather than corrected, because the design cannot be undone after the fact.

## Every measured quantity

| Quantity | Local M0 | GPT, base model | Claude Opus, high effort | Gemini Pro |
|---|---|---|---|---|
| Draws | 1,550 | 20 | 20 | 20 |
| Executed and took a position | 225 (14.5%) | 20/20 (100%) | 20/20 (100%) | 20/20 (100%) |
| Ruined mid-window | — | 1 | 0 | 1 |
| Static rejected | 222/1,550 = 14.3%; 28/225 rankable = 12.4% | **0/20** | **0/20** | **1/20** |
| Semantic rejected | — | 2/20 | 6/20 | 5/20 |
| Statistical rejected | — | 20/20 | 20/20 | 20/20 |
| PBO | — | 0.0190 | 0.0378 | 0.0037 |
| Exact duplicate clusters | 11 clusters (n = 156) | 4 over 11/19 | 3 over 6/20 | 2 over 5/19 |
| Near-duplicate pairs > 0.9999 | — | 0 | 3 | 0 |
| Fragility, median | 0.360 (n = 125) | 0.418 | 0.355 | 0.334 |
| Fragility, min–max | — | 0.189–32.785 | 0.169–1.373 | 0.164–2.703 |
| Mean regime Sharpe near zero | 3 of 125 | 1 | 0 | 0 |
| Binding capacity, median | Rs 2.60 cr (n = 156) | Rs 2.99 cr | Rs 5.12 cr | Rs 0.80 cr |
| Capacity ratio to M0 | 1.00x | **1.15x** | **1.97x** | **0.31x** |
| Capacity, min–max | — | Rs 0.38–64.28 cr | Rs 0.38–26.33 cr | Rs 0.19–67.31 cr |
| Capacity span | 30.9x | 167.0x | 68.4x | 349.7x |

## Deflated Sharpe, both readings

Amendment 2 fixes per-arm deflation plus a matched-size comparison against M0, and
forbids quoting either alone. It says an arm is deflated at *its own trial count*
and illustrates that with N = 5, the arm size expected when it was written; the arms
as collected are 20. **Both readings are published**, per the PI's ruling.

| Arm | N | Clearing DSR >= 0.95 | Matched M0 draws clearing | Empirical p |
|---|---|---|---|---|
| GPT, base model | 5 | 0/20 | 41/1000 | 0.041 |
| GPT, base model | 20 | 0/20 | 3/1000 | 0.003 |
| Claude Opus, high effort | 5 | 0/20 | 41/1000 | 0.041 |
| Claude Opus, high effort | 20 | 0/20 | 3/1000 | 0.003 |
| Gemini Pro | 5 | 0/20 | 41/1000 | 0.041 |
| Gemini Pro | 20 | 0/20 | 3/1000 | 0.003 |

The matched figures repeat across arms because the subsampling depends only on the
draw size and the fixed seed, not on which arm it is compared against. That is
correct, not a duplicated row.

## The holdout — evaluated once, for the whole study

2025-01-01 to 2025-12-31, never seen during development or during any methodology
decision in this study. Authorised by the PI on 2026-08-04 after all three arms
were collected, under RULE 7's amendment, whose three conditions were verified in
writing from git history beforehand. **No tuning of anything follows this table.**

| Arm | Dev Sharpe, mean | Holdout mean | Holdout median | Holdout best | Best DSR |
|---|---|---|---|---|---|
| GPT, base model | -0.1394 | -0.9491 | -1.1492 | +0.3376 | 0.0000 |
| Claude Opus, high effort | -0.1635 | -0.5781 | -0.7696 | +0.4419 | 0.0000 |
| Gemini Pro | -0.5854 | -1.4768 | -1.1492 | +0.3827 | 0.0000 |

M0's own 225 rankable strategies score a mean holdout Sharpe of **-1.0351** (P1
`RESULTS.md`). Every frontier arm lands in the same territory: negative on average,
with a best case near zero.

| Arm | N | Clearing DSR >= 0.95 | Matched M0 draws clearing | Empirical p |
|---|---|---|---|---|
| GPT, base model | 5 | 0/20 | 0/1000 | 0.000 |
| GPT, base model | 20 | 0/20 | 0/1000 | 0.000 |
| Claude Opus, high effort | 5 | 0/20 | 0/1000 | 0.000 |
| Claude Opus, high effort | 20 | 0/20 | 0/1000 | 0.000 |
| Gemini Pro | 5 | 0/20 | 0/1000 | 0.000 |
| Gemini Pro | 20 | 0/20 | 0/1000 | 0.000 |

**Not one of the 60 frontier strategies clears DSR >= 0.95 on the holdout, at either
N.** Neither does any of the 1,000 matched M0 subsamples, at either N --- on the
holdout the local corpus clears 0/1000 where on development data it cleared 3/1000.
The bar is not merely un-cleared by the frontier arms; it is un-cleared by everything.

**H6 is confirmed on both halves.** The pre-registered prediction was that frontier
generators would not change the study's statistical conclusions, and they did not.

## The five measurements added after a coverage audit

Every benchmark makes more than one measurement, and the first pass of this study put the
arms through each benchmark's *headline* only. The audit that found this was prompted by
the PI, not by us. All five are reported below, including the two that correct figures
stated earlier in this study's own history.

| Quantity | Local M0 | GPT, base model | Claude Opus, high effort | Gemini Pro |
|---|---|---|---|---|
| Fragility across regimes, median (P2 **primary**) | 0.618 (n = 125) | 0.566 | 0.624 | 0.696 |
| Fragility across regimes, range | --- | 0.189--3.202 | 0.043--1.094 | 0.048--2.332 |
| Capacity, outflow / inflow, median | 0.96 (5 factors) | 0.952 | 0.868 | 0.905 |
| Capacity ratio, range | 0.94--1.24 | 0.770--1.041 | 0.700--1.116 | 0.615--1.414 |
| Knife-edge under a 9e-15 panel change | 31/156 = 19.9% | **3/20** | **7/20** | **4/20** |
| Nondeterministic across hash seeds | 27/156 = 17.3% | **0/20** | **0/20** | **0/20** |
| Mean Sharpe lost to costs | 0.5867 | 0.5004 | 0.5795 | 0.8120 |
| Profitable gross, unprofitable net | 35/225 = 15.6% | 6/20 | 1/20 | 4/20 |

**Two of these correct earlier statements in this study.** The knife-edge row was
previously reported as zero for two arms; that figure came from reading
`mean_is_near_zero`, which is a different test. Run properly, the pathology recurs on
every arm at a rate comparable to the local corpus. Determinism was asserted before it
was tested; the assertion held, but it was untested when made.


## The abstention frontier --- exploratory

AUAP is AlphaAudit's primary metric and appears nowhere in this study's
pre-registration. It was computed only after the PI asked for it, from holdout return
series already spent, and **every figure here is exploratory**. Coverage granularity is
one strategy in twenty, against one in 225 locally.

| Auditor layers | Local M0 (n = 225) | GPT, base model | Claude Opus, high effort | Gemini Pro |
|---|---|---|---|---|
| semantic | -1.1967 | -0.9717 | -0.6083 | -1.7755 |
| semantic + statistical | -1.3777 | -1.3976 | -0.6083 | -1.7755 |
| static | -1.2529 | -1.0348 | -0.6318 | -1.7411 |
| static + semantic | -1.2782 | -0.9717 | -0.6083 | -1.7845 |
| static + semantic + statistical | -1.4184 | -1.3976 | -0.6083 | -1.7845 |
| static + statistical | -1.4339 | -1.4407 | -0.6318 | -1.7411 |
| statistical | -1.3497 | -1.4407 | -0.6318 | -1.7487 |
| *random 95% interval* | *[-1.221, -0.860]* | [-1.481, -0.532] | [-0.871, -0.300] | [-2.248, -0.858] |

**No layer combination beats random rejection on any arm** --- 0 of 21 cells. The null AlphaAudit published on its own corpus
replicates on all three frontier populations.

Two structures carry across every population. Adding the statistical layer makes
selectivity *harmful*: the most-trusted single strategy is worse than the average one.
And the static layer contributes no ordering at all --- on the Claude arm, every
combination containing it is identical to the same combination without it, to four
decimal places, because it flags almost nothing and so cannot reorder anything.


## The fragility predictor, out of population --- exploratory

RegimeStress trained a model mapping strategy characteristics onto fragility and
reported that it does not work: an out-of-sample R-squared of +0.024 against a mean
baseline. The model, its features and its seed are unchanged here; the frontier arms are
pure held-out data. **Not pre-registered.**

| Arm | n | R2 vs training mean | Spearman rho | MAE model / baseline |
|---|---|---|---|---|
| gpt | 20 | +0.122 | +0.717 | 3.264 / 2.872 |
| claude | 20 | -0.445 | +0.540 | 1.154 / 1.314 |
| gemini | 20 | -2.663 | +0.556 | 1.388 / 1.417 |
| **pooled** | 60 | **-0.004** | **+0.555** | 1.936 / 1.868 |

**The level predictions are worthless and the negative result survives**: a pooled
R-squared of essentially zero means the model does no better than predicting its own
training mean on a population it never saw.

**The rank ordering is not worthless**, which was not expected. Spearman is positive on
all three arms independently and 0.555 pooled. The model cannot say how fragile a
strategy is, but it partially orders which is more fragile. Pooling across arms is not
exchangeable and n = 20 per arm is thin, so this is a direction worth testing properly,
not a result.


## Auditor detail, including the layers that found nothing

| Arm | Static classes raised | Semantic labels |
|---|---|---|
| GPT, base model | **none** | `consistent` x18, `rationale_implementation_mismatch` x2 |
| Claude Opus, high effort | **none** | `consistent` x14, `rationale_implementation_mismatch` x5, `unacknowledged_known_anomaly` x1 |
| Gemini Pro | `snooped_parameter` x1 | `consistent` x15, `rationale_implementation_mismatch` x5 |

**The static layer raised 1 finding across 60 frontier
strategies**, against 12.4% of M0's rankable candidates. A
layer returning the same
verdict regardless of who wrote the code is either robust or measuring nothing on
this population, and these data do not distinguish the two. That is RQ4, and it is
reported unresolved.

## Hypotheses as pre-registered

| | Pre-registered claim | Outcome |
|---|---|---|
| H1 | Executability rises sharply; rankable rate >= 40% | **Confirmed, 3 of 3** — 100% against M0's 14.5% |
| H2 | Audit pass rate does not improve | **Falsified, 3 of 3** — 1 static finding in 60 |
| H3 | The blind spot is `full_sample_statistic` | **Not supported** — the single finding was `snooped_parameter` |
| H4 | Diversity does not improve | **Confirmed, 3 of 3** — every arm duplicates across independent requests |
| H5 | Capacity within 2x of M0 | **Falsified on Gemini Pro** at 0.31x; held on the other two |
| H6 | No frontier strategy clears deflation | **Confirmed on both halves, 3 of 3** — 0 of 60 at either N, development and holdout |

## What these results do not establish

- **Why Gemini Pro's capacity is a third of M0's.** Not investigated. Any account
  would be exploratory, and is deliberately absent rather than guessed at.
- **Whether the static layer is robust or inert on frontier code.** One finding in
  60 cannot separate those.
- **Whether 20 draws per arm is enough.** It is not enough for a tight interval on
  any per-arm rate, and the pooled-request design makes the effective sample smaller.
- **Anything about models other than these three, at these settings, on this date.**

