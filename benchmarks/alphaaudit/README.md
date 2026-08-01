# AlphaAudit — a benchmark for auditing AI-generated trading strategies

Measures whether an auditor can tell, in advance and out of sample, which machine-generated
strategies are worth acting on. The headline metric is **AUAP** — the area under the
abstention–performance curve — scored against random rejection at matched coverage.

---

## The evaluation protocol

**This is a frozen benchmark scored once per submission. The protocol is the benchmark.**

1. **The holdout period is fixed and shared.** Every submission is scored on the same held-out
   window under the same cost model, universe construction and execution assumptions.
2. **One evaluation per submission.** A submission is scored on the holdout exactly once.
3. **No tuning on holdout results.** Nothing — auditor, generator, prompt, thresholds, or
   post-processing — may be adjusted after seeing a holdout number. If you change anything, that is
   a new submission and it needs a new one-shot evaluation.
4. **The benchmark is not modified in response to submissions.** The holdout window, the metric and
   the baseline are not revised because a result was disappointing or surprising.

Multiple independent one-shot evaluations do not contaminate a frozen test set — iterative
optimisation against it does. That distinction is the whole of the discipline here, and the
benchmark's comparability claim rests entirely on entrants honouring rule 3. It is stated plainly
rather than assumed so that "results are comparable" is a real claim and not an aspiration.

**Report every AUAP with its trial count `N` and its generator.** An AUAP without the trial count
that produced its corpus is not interpretable.

---

## Reference implementation: Qwen-7B

The first entry, provided so the harness has a worked example end to end — **not** a claim about
what auditing can achieve in general.

| Field | Value |
|---|---|
| Generator | `qwen2.5:7b-instruct-q4_K_M`, temperature 0.8, local via Ollama |
| Prompt digest | `f307433c7bda8595` |
| Corpus | 1,550 candidates |
| Executed | 631 (40.7%) |
| Flat — executed, never traded | 406 (64.3% of executed) |
| Rankable | 225 (14.5% of corpus) |
| Trial count `N` | 1,887 |
| Auditor layers | static (AST), semantic (local LLM), statistical (DSR + PBO) |
| Performance basis | **net of Indian transaction costs**, Rs. 10,00,000 book, retail depository mode |
| Best holdout AUAP | **-1.1441** (semantic alone) vs random 95% interval **[-1.2208, -0.8600]** |
| Configurations beating random | **0 of 7** |

### Two conditions that travel with this number permanently

**1. AUAP is generator-dependent.** It is computed over a ranked corpus, and that corpus came from
one specific weak generator. It is not a generator-independent property of the auditor, and it must
never be quoted as "the AlphaAudit result" — only as the result for this reference implementation.

**2. A null result here is confounded with corpus degeneracy.** The 225 rankable strategies are
tightly clustered and mostly weak, with little dispersion for any ranking to exploit. **A null AUAP therefore cannot distinguish "the auditor is uninformative" from "this
corpus is too weak to test the auditor."** Both explanations survive the data.

That confound is the motivation for other entrants rather than a defect in the benchmark. A stronger
generator producing genuinely dispersed strategy quality is exactly what would let AUAP discriminate
— which is the case for running this harness against better models.

---

## Contents

| Path | What it is |
|---|---|
| `RESULTS.md` | **every number in the project, with its sample size** — start here |
| `survivors/` | 174 strategies cleared by the static and semantic layers — the P1 → P2 bridge set |
| `../../tests/fixtures/leaky/` | deliberately-cheating strategies; positive controls for the auditor |
| `../../tests/fixtures/clean/` | honest strategies; measure the false-positive rate |
| `../../tests/fixtures/locked/` | held-out fixture set, scored exactly once |
| `static_blind_spots.md` | documented failure modes of the static analyser |

## Reproducing

```bash
python scripts/run_generation.py --n 300 --start-index 0
python scripts/run_corpus_backtest.py --corpus runs/<stamp>/candidates
python scripts/run_corpus_audit.py --corpus runs/<stamp>/candidates
python scripts/pool_corpora.py
python scripts/run_ablation.py --corpus runs/pooled/candidates
python scripts/plot_abstention.py --ablation runs/pooled/ablation_development.json
```

The holdout path requires explicit authorisation and is gated structurally:

```bash
python scripts/run_ablation.py --corpus runs/pooled/candidates --holdout --authorised-by "<who, when>"
```

Without `--authorised-by` the script exits 2 rather than reading the holdout.
