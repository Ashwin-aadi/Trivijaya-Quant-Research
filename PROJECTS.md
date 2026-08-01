# The three projects, and where each one lives

This lab is one repository holding three projects that run in sequence. Each answers the question
the previous one raises.

```
P1  ALPHAAUDIT     Is this signal real, or is it leakage and luck?      COMPLETE
        |  survivors + a standard factor panel pass to
P2  REGIMESTRESS   When does it break?                                  RELEASED, NOT STARTED
        |  robust strategies pass to
P3  FLOWSTATE      How much money can it absorb?                        NOT STARTED
```

This file exists because the repository is **not** laid out as three sibling folders, and that is
deliberate. Roughly half of what P1 built is shared infrastructure that P2 and P3 are *required* to
reuse — the same cost model, the same backtester, the same survivorship-free universe. Copying it
into a per-project folder would mean three drifting copies of the cost model, and the moment they
disagree, cross-project comparisons stop meaning anything. Sharing it is the reason the sequence is
a pipeline rather than three unrelated repositories.

So instead of moving files, this is the map.

---

## Shared infrastructure — owned by no single project

Built during P1, but part of the lab rather than part of AlphaAudit. Changing anything here
affects results already published in P1, so changes need PI approval.

| Path | What it is |
|---|---|
| `src/common/` | config loader, structured logging, seeding, run manifests, `PointInTimeError` |
| `src/data/` | NSE calendar, point-in-time survivorship-free universe, prices, corporate actions |
| `src/costs/` | Indian statutory cost schedule, slippage, impact, tradability constraints |
| `src/backtest/` | point-in-time engine, strategy interface, purged k-fold CV with embargo |
| `src/eval/` | metrics, plots, reporting |
| `config/config.yaml` | every tunable parameter for every project |
| `env/` | setup, dependency pins, hardware report |
| `data/` | immutable raw NSE data plus regenerable derivatives (local, not tracked) |
| `runs/` | one directory per execution, each with a manifest (local, not tracked) |

Shared scripts: `download_bhavcopy.py`, `build_universe.py`, `build_corporate_actions.py`,
`cross_check_prices.py`, `build_holdout.py`, `show_artifacts.py`, `progress.py`.

---

## P1 — AlphaAudit

**Status: complete.** Checkpoint 1.5 approved; all five questions answered. The holdout is
permanently closed after two authorised evaluations.

Landing page: [`benchmarks/alphaaudit/README.md`](benchmarks/alphaaudit/README.md) ·
Every number: [`benchmarks/alphaaudit/RESULTS.md`](benchmarks/alphaaudit/RESULTS.md) ·
Paper: [`papers/alphaaudit.pdf`](papers/alphaaudit.pdf)

| Path | What it is |
|---|---|
| `src/audit/` | the three auditor layers — `static.py` (AST), `stat.py` (DSR + PBO), `semantic.py` (local LLM) |
| `src/generate/` | LLM strategy generator, constrained to the engine's interface |
| `benchmarks/alphaaudit/` | the frozen benchmark: protocol, results, survivors, human labels, blind spots |
| `tests/audit/` | auditor tests |
| `tests/fixtures/leaky/` | strategies that cheat on purpose — positive controls |
| `tests/fixtures/clean/` | honest strategies — false-positive controls |
| `tests/fixtures/locked/` | held-out fixture set, scored exactly once |
| `tests/fixtures/refine/` | fixtures used while iterating, before the lock |
| `papers/alphaaudit.tex` | the write-up |

P1 scripts: `run_generation.py`, `run_corpus_backtest.py`, `run_corpus_audit.py`,
`run_stat_audit.py`, `run_semantic_kappa.py`, `run_ablation.py`, `plot_abstention.py`,
`pool_corpora.py`, `tag_survivors.py`, `build_label_sheet.py`, `run_fixtures.py`,
`run_positive_control.py`, `attribution_report.py`, `gross_vs_net.py`, `dp_sensitivity.py`,
`decompose_tracking_error.py`, `run_net_pipeline.sh`, `resume_net_pipeline.sh`.

> The holdout steps inside `run_net_pipeline.sh` carry a **spent** authorisation and must not be
> re-run.

---

## P2 — RegimeStress

**Status: released at Checkpoint 1.5, not started.** Phase 2.0 has its own halt condition on
regime-label leakage that must reach the PI as a decision before any labelling code is written.

Landing page: [`benchmarks/regimestress/README.md`](benchmarks/regimestress/README.md)

| Path | What it will be |
|---|---|
| `src/stress/` | regime labelling, counterfactual regime resampling, fragility measurement |
| `benchmarks/regimestress/` | the frozen P2 benchmark |
| `papers/regimestress.tex` | the write-up |

None of these exist yet. They are listed so the shape is visible, not as a claim that they are
present.

---

## P3 — FlowState

**Status: not started, and not released.** Do not begin until the PI explicitly releases P2 at
Checkpoint 2.3.

| Path | What it will be |
|---|---|
| `src/capacity/` | flow ingestion, impact models, alpha decay, capacity curves |
| `benchmarks/flowstate/` | the frozen P3 benchmark |
| `papers/flowstate.tex` | the write-up |

---

## Two things this file is not

**It is not a substitute for the charter.** `CLAUDE.md` is the specification and is kept off the
remote along with `DECISIONS.md` and `reports/`. This file is a map of what exists.

**It is not a claim that the projects are independent.** P2 consumes P1's survivor set, P1's cost
model, and P1's universe. A change to the cost model changes P1's published numbers *and* P2's
inputs simultaneously. That coupling is the point, and it is why the shared layer is not duplicated
per project.
