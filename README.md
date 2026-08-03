# Trivijaya Quant Research

**Measurement infrastructure for quantitative research on Indian equities.**

This repository builds instruments that detect when a quantitative result is false.

Almost every trading signal that looks profitable on historical data is not. The two dominant
causes are **leakage** — the code sees the future — and **multiple testing** — try enough things
and something looks brilliant by luck. This is the apparatus that separates real findings from
artifacts, applied to a market where it does not currently exist openly.

A negative result is a successful outcome here. *"The signal did not survive"* is the honest
majority case, and the repository is built to report it rather than to avoid it.

---

## One suite, three questions

The three benchmarks are the sequence a systematic fund uses to take an idea from *interesting* to
*we are allocating capital to this*. Each answers the question the previous one raises.

```
AlphaAudit     →  Is this signal real, or is it leakage and luck?
      ↓  survivors pass to
RegimeStress   →  When does it break?
      ↓  survivors pass to
FlowState      →  How much money can it absorb, and how fast does the edge die?
```

All three are **released and frozen**. Each is versioned, reproducible from raw data by a
documented command sequence, and carries a `CORRECTIONS.md` recording every defect found after
freezing and exactly what it changed.

**125 strategies now carry all three verdicts** — an audit result, a fragility score and a
deployment capacity apiece. That corpus is the point of building the three together: as far as we
are aware, no other public collection of Indian equity strategies is scored on all three axes.

[`PROJECTS.md`](PROJECTS.md) maps which paths belong to which project, and which are shared
infrastructure.

---

## AlphaAudit — is the signal real?

> **A Benchmark for Auditing AI-Generated Trading Strategies**
> Paper: [`papers/alphaaudit.pdf`](papers/alphaaudit.pdf) · Every number:
> [`benchmarks/alphaaudit/RESULTS.md`](benchmarks/alphaaudit/RESULTS.md)

A local 7B model wrote 1,550 candidate trading strategies. A three-layer auditor — AST leakage
analysis, statistical deflation, and a semantic check of stated rationale against implemented code
— tried to prove they were fooling themselves. Everything was measured net of Indian transaction
costs, against a survivorship-free universe, on a holdout evaluated three times under logged
authorisation and now permanently closed.

| | |
|---:|---|
| **1,550** | candidates generated, all parseable and interface-conforming |
| **40.7%** | survived contact with real data |
| **14.5%** | both executed *and* traded |
| **15.6%** | of profitable strategies reversed sign once costs were charged |
| **11** | were bankrupted outright by costs |
| **0 / 631** | cleared deflation at an honest trial count of *N* = 1,887 |
| **28 / 222** | leak flags landed on code that actually trades |
| **0 / 7** | auditor configurations beat random rejection |

**Three findings, in order of how much they surprised us:**

1. **The dominant failure mode of an LLM strategy researcher is not that it cheats — it is that
   it does not run.** Six candidates in ten never survived contact with data; two thirds of the
   rest never took a position. An auditor tuned to catch leakage guards a door most candidates
   never reach.
2. **Indian transaction costs, honestly modelled, are large enough to invert conclusions.** Two of
   eleven standard academic factors flip from profitable to unprofitable, and momentum at a net
   Sharpe of 0.966 no longer beats equal-weighting the same universe.
3. **The auditor did not beat random rejection**, in any of seven layer configurations, out of
   sample. This null survived a full correction of our own cost implementation: the magnitudes
   moved, the conclusion did not. It remains confounded with corpus degeneracy, and that caveat
   travels with it permanently.

---

## RegimeStress — when does it break?

> **Counterfactual Regime Resampling and Learned Fragility Estimation**
> Paper: [`papers/regimestress.pdf`](papers/regimestress.pdf) · Every number:
> [`benchmarks/regimestress/RESULTS.md`](benchmarks/regimestress/RESULTS.md) · Frozen as
> `regimestress-v1`

A strategy that performed well over the development window was tested against **one** sequence of
history. Indian markets since 2015 contain approximately one pandemic crash, one full rate cycle
and one structural liquidity shift, so the number of independent regime observations is very
small. RegimeStress labels market conditions causally, resamples plausible histories that never
happened, and measures how much each strategy's performance moves across them.

| | |
|---:|---|
| **4** | HMM states, selected by BIC on pre-development data and then frozen permanently |
| **20** | expanding-window refits — no label uses information from after the session it describes |
| **21.8%** | of labels disagree with what a full-sample fit would have assigned |
| **158 / 185** | strategies deterministic enough to stress at all |
| **31** | excluded as knife-edge: a $10^{-15}$ perturbation changes their behaviour |
| **+0.212** | best out-of-fold R² for the fragility predictor |
| **0.005** | permutation p-value for rank, against 0.861 for level on three of four targets |

**Three findings:**

1. **Regime labels are not innocent.** A label fitted on the full sample and then used to partition
   strategy performance is leakage of the same species the suite exists to detect. Fitted causally,
   more than a fifth of the labels differ.
2. **Fragility is learnable in rank and not in level.** Which strategies are more fragile than
   which is predictable from strategy characteristics; *how* fragile any one of them is, is not.
3. **Duplicate strategies inflated the predictor by 0.238 of R² on their own** — a twin sitting in
   a training fold while its pair is scored in the test fold. The earlier feature ranking was
   substantially an artefact of that, and it changed completely once duplicates were removed.

---

## FlowState — how much can it take?

> **Flow-Conditional Capacity and Alpha-Decay Estimation in a Retail-Dominated Market**
> Paper: [`papers/flowstate.pdf`](papers/flowstate.pdf) · Every number:
> [`benchmarks/flowstate/RESULTS.md`](benchmarks/flowstate/RESULTS.md) · Frozen as `flowstate-v1`

The first question a portfolio manager asks is how much money a strategy can take. The standard
answer requires knowing how far a given order moves the price. **We measured whether daily bars can
supply that, and they cannot** — so FlowState reports what daily bars *do* support: capacity as a
**constraint**, the largest AUM at which every trade fits inside a stated participation limit. That
is arithmetic on observed turnover and assumes no impact coefficient.

| | |
|---:|---|
| **₹0.81–7.91 cr** | deployment capacity of standard price-and-volume factors |
| **3 of 4** | horizons at which two defensible weighting schemes disagree on the *sign* of impact |
| **0.836 vs 0.374** | split-sample stability of Amihud illiquidity vs the fitted impact exponent, n = 166 symbols |
| **2 / 208,042** | observations in the participation region a capacity study asks about |
| **0** | factors with a significant edge at the shortest horizon |
| **10.6×** | by which the median session overstates the typical corpus strategy |
| **125** | strategies carrying all three benchmark results |

**Four findings:**

1. **Daily OHLCV cannot identify a transient-impact coefficient.** Not for want of precision: two
   equally defensible estimators disagree about its sign and nothing in the data adjudicates. A
   control on the same panel — Amihud illiquidity — is perfectly stable, so the failure is in the
   model rather than the data. A great many published backtests price impact from daily data.
2. **Capacity is set by the least liquid position a strategy holds**, not by the average liquidity
   of its book, because equal weighting forces the same rupee size into a ₹50 crore-a-day name as
   into a ₹500 crore-a-day one.
3. **Deployable size does not collapse when foreign participation reverses** — a null, and one
   conditional on a derivatives-activity proxy we would rather not have needed.
4. **A benchmark's soundness cannot be established on the population that motivated it.** FlowState
   was validated on 5 factor strategies spanning 9.8× in capacity, then applied to 156
   machine-generated ones spanning 30.9×. Two defects lived in that gap and the corpus found both
   immediately. The corrections moved every published figure **down**, by between 4.3× and 26×.

That last finding changed how this lab works. The standing order is now **build → validate on
standard strategies → apply to the reference corpus → freeze → write up**. The corpus run precedes
the freeze and the paper, because a validation set assembled from familiar, well-behaved strategies
has no tails, and defects live in tails.

---

## What's in here

| Path | What it is |
|---|---|
| `src/data/` | NSE trading calendar, **survivorship-free universe**, prices, corporate actions |
| `src/costs/` | Indian transaction cost model — statutory schedule, slippage, tradability constraints |
| `src/backtest/` | Point-in-time engine that *raises* on a stale signal; purged k-fold CV with embargo |
| `src/audit/` | The three auditor layers: AST leakage, statistical deflation, semantic |
| `src/generate/` | LLM strategy generator, constrained to the engine's interface |
| `src/stress/` | Causal regime labelling, counterfactual regime resampling, fragility |
| `src/capacity/` | Flow states, liquidity measures, alpha decay, constraint-based capacity |
| `src/eval/` | Metrics, plots, reporting |
| `benchmarks/alphaaudit/` | Fixtures, survivors, human labels, protocol, results |
| `benchmarks/regimestress/` | Regime labels, stress paths, fragility scores, results |
| `benchmarks/flowstate/` | Capacity curves, decay curves, the reference corpus, results |
| `tests/fixtures/leaky/` | Strategies that cheat **on purpose** — the positive controls |
| `tests/fixtures/clean/` | Honest strategies — the false-positive controls |
| `papers/` | The three write-ups and their generated figures |
| `config/config.yaml` | Every tunable parameter. No magic numbers live in source. |

`data/` and `runs/` are generated locally and not tracked — everything in them is regenerable from
documented commands.

---

## Four design decisions worth knowing about

**The universe is survivorship-free by construction, and is not the NIFTY 100.** Free
point-in-time index membership could not be sourced, so the universe is rules-based: at each
quarterly rebalance, the 100 NSE equities with the highest trailing 126-session median traded
value, computed using only sessions strictly *before* that date. Nothing consults a membership
list, so a company that later collapses simply stops qualifying rather than vanishing from
history. 185 distinct names appear across 20 rebalances; 41 of the first cohort had left by the
last. Those 41 are the evidence the construction works.

**The engine makes lookahead hard to express, not merely discouraged.** A strategy never receives
the price panel — only a view already truncated to before its decision moment. Every signal
carries an `information_available_at` stamp, and the engine raises `PointInTimeError`, not a
warning, if that stamp is not strictly prior to the fill session.

**The trial ledger sits outside the generator's write path.** Deflated Sharpe is meaningless
without an honest *N*, and a generator that emits a candidate every few seconds destroys a
researcher's ability to know their own denominator. Every draw increments a hash-chained,
append-only ledger — including failures and retries — and verification walks the chain and raises
on any broken link.

**No number in a paper is typed by hand.** Every figure in the RegimeStress and FlowState papers is
a LaTeX macro generated from a frozen artifact. `scripts/check_paper_numbers.py` fails the build if
a paper uses a macro nothing defines, defines one nothing uses, or contains a bare numeral in a
claim position. A test takes the committed paper, substitutes one macro for the value it renders
to, and requires the checker to start failing — because a gate nobody has seen fail is
indistinguishable from a gate that cannot fail.

---

## Setup

```bash
bash env/setup.sh                 # venv, dependencies, Ollama check, hardware report
source .venv/Scripts/activate     # Windows (Git Bash); .venv/bin/activate on POSIX
```

All inference runs locally through [Ollama](https://ollama.com). **There are zero paid API calls
anywhere in this repository.** Default models: `qwen2.5:7b-instruct-q4_K_M` for reasoning and
semantic audit, `qwen2.5-coder:14b-instruct-q4_K_M` for code generation.

## Verifying it yourself

Start here rather than with the code. These are the checks that fail loudly if something is wrong.

```bash
# The cost model, pinned to a figure verified against a discount broker's calculator
python -m pytest tests/costs/test_india.py -q

# Deliberate cheats must be caught; honest strategies must not be flagged
python -m pytest tests/audit/test_static.py -q

# A strategy trading on a stale signal must raise, not warn
python -m pytest tests/backtest/test_engine.py -q -k point_in_time

# No paper may state a figure that did not come from an artifact
python scripts/check_paper_numbers.py

# Everything
python -m pytest -q && ruff check . && mypy
```

## Rebuilding the data

```bash
python scripts/download_bhavcopy.py        # NSE end-of-day archives -> data/raw/ (immutable)
python scripts/build_universe.py           # survivorship-free universe
python scripts/build_corporate_actions.py  # split/bonus adjustment
```

`data/raw/` is write-once; every file carries a `.meta.json` with its source URL, fetch timestamp,
row count and SHA256. Everything downstream is regenerable.

## Reproducing each benchmark

```bash
# AlphaAudit — the full chain, net of costs
bash scripts/run_net_pipeline.sh
python scripts/gross_vs_net.py --results runs/pooled/backtest_results.json

# RegimeStress — causal regime labels, then the stress suite
python scripts/select_regime_states.py     # BIC state selection, pre-development data only
python scripts/build_regimes.py            # expanding-window labels
python scripts/deduplicate_corpus.py       # exact and near-duplicate detection
python scripts/run_stress_tier1.py && python scripts/run_stress_tier2.py

# FlowState — decay, capacity, and the corpus run
python scripts/fetch_participant_flows.py
python scripts/build_flowstate.py
python scripts/diagnose_impact_identifiability.py
python scripts/run_corpus_capacity.py
```

**The holdout steps in the AlphaAudit script carry a spent authorisation and must not be re-run.**
The holdout was evaluated three times, each under written authorisation, and is permanently closed.

---

## Standards

Python 3.11+ · `polars` over `pandas` · full type hints · `ruff` + `mypy` clean · `pytest` ·
global seed 42, every stochastic operation explicitly seeded · a run manifest written for every
execution, carrying git SHA, config hash, package versions, model tags and input hashes.

`benchmarks/alphaaudit/survivors/` and `tests/fixtures/locked/` are exempt from linting. They hold
machine-generated source exactly as emitted, and the published leak-class counts are a measurement
of those precise bytes. Reformatting them would destroy the artifact they are evidence of.

## Citing

```bibtex
@misc{jain2026alphaaudit,
  author = {Ashwin Jain},
  title  = {AlphaAudit: A Benchmark for Auditing AI-Generated Trading Strategies,
            with a Reference Implementation and Negative Result on Indian Equities},
  year   = {2026}
}

@misc{jain2026regimestress,
  author = {Ashwin Jain},
  title  = {RegimeStress: Counterfactual Regime Resampling and Learned Fragility
            Estimation for Indian Equity Strategies},
  year   = {2026}
}

@misc{jain2026flowstate,
  author = {Ashwin Jain},
  title  = {FlowState: Constraint-Based Deployment Capacity and Alpha Decay in Indian
            Equities, and Why Daily Bars Cannot Price Market Impact},
  year   = {2026}
}
```

## License

Not yet specified.
