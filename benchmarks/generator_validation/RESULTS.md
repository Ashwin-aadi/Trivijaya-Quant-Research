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
| Static rejected | 222/1,550 = 14.3%; 26/225 rankable = 11.6% | **0/20** | **0/20** | **1/20** |
| Semantic rejected | — | 2/20 | 6/20 | 5/20 |
| Statistical rejected | — | 20/20 | 20/20 | 20/20 |
| PBO | — | 0.0190 | 0.0378 | 0.0037 |
| Exact duplicate clusters | 11 clusters (n = 156) | 4 over 11/19 | 3 over 6/20 | 2 over 5/19 |
| Near-duplicate pairs > 0.9999 | — | 0 | 3 | 0 |
| Fragility, median | 0.360 (n = 125) | 0.418 | 0.355 | 0.334 |
| Fragility, min–max | — | 0.189–32.785 | 0.169–1.373 | 0.164–2.703 |
| Mean-near-zero flagged | 31 knife-edge of 156 | 1 | 0 | 0 |
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

## Auditor detail, including the layers that found nothing

| Arm | Static classes raised | Semantic labels |
|---|---|---|
| GPT, base model | **none** | `consistent` x18, `rationale_implementation_mismatch` x2 |
| Claude Opus, high effort | **none** | `consistent` x14, `rationale_implementation_mismatch` x5, `unacknowledged_known_anomaly` x1 |
| Gemini Pro | `snooped_parameter` x1 | `consistent` x15, `rationale_implementation_mismatch` x5 |

**The static layer raised 1 finding across 60 frontier
strategies**, against 11.6% of M0's rankable candidates. A layer returning the same
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

