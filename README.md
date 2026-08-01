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

## Project 1 — AlphaAudit: complete

> **A Benchmark for Auditing AI-Generated Trading Strategies**
> Paper: [`papers/alphaaudit.pdf`](papers/alphaaudit.pdf) · Every number:
> [`benchmarks/alphaaudit/RESULTS.md`](benchmarks/alphaaudit/RESULTS.md)

A local 7B model wrote 1,550 candidate trading strategies. A three-layer auditor tried to prove
they were fooling themselves. Everything was measured net of Indian transaction costs, against a
survivorship-free universe, on a holdout evaluated twice under logged authorisation and now
permanently closed.

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

Projects 2 (RegimeStress) and 3 (FlowState) are not yet started.

---

## What's in here

| Path | What it is |
|---|---|
| `src/data/` | NSE trading calendar, **point-in-time survivorship-free universe**, prices, corporate actions |
| `src/costs/` | Indian transaction cost model — statutory schedule, slippage, impact, tradability constraints |
| `src/backtest/` | Point-in-time engine that *raises* on a stale signal; purged k-fold CV with embargo |
| `src/audit/` | The three auditor layers: AST leakage, statistical deflation, semantic |
| `src/generate/` | LLM strategy generator, constrained to the engine's interface |
| `src/eval/` | Metrics, plots, reporting |
| `benchmarks/alphaaudit/` | The frozen benchmark: fixtures, survivors, human labels, protocol, results |
| `tests/fixtures/leaky/` | Strategies that cheat **on purpose** — the positive controls |
| `tests/fixtures/clean/` | Honest strategies — the false-positive controls |
| `papers/` | The write-up and its figures |
| `config/config.yaml` | Every tunable parameter. No magic numbers live in source. |

`data/` and `runs/` are generated locally and not tracked — everything in them is regenerable from
documented commands.

---

## Three design decisions worth knowing about

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

# Everything
python -m pytest -q && ruff check . && mypy src tests
```

## Rebuilding the data

```bash
python scripts/download_bhavcopy.py        # NSE end-of-day archives -> data/raw/ (immutable)
python scripts/build_universe.py           # point-in-time universe
python scripts/build_corporate_actions.py  # split/bonus adjustment
```

`data/raw/` is write-once; every file carries a `.meta.json` with its source URL, fetch timestamp,
row count and SHA256. Everything downstream is regenerable.

## Reproducing the Project 1 results

```bash
bash scripts/run_net_pipeline.sh           # the full chain, net of costs
python scripts/gross_vs_net.py --results runs/pooled/backtest_results.json
python scripts/dp_sensitivity.py           # cost sensitivity to assumed book size
```

**The holdout steps in that script carry a spent authorisation and must not be re-run.** The
holdout was evaluated twice, both under written authorisation, and is permanently closed.

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
```

## License

Not yet specified.
