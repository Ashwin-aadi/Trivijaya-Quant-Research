# FlowState v1

**A frozen benchmark for deployment capacity and alpha decay in Indian equities, measured from
daily bars alone.**

Third of three benchmarks in this programme. AlphaAudit asks whether a strategy is real;
RegimeStress asks when it breaks; FlowState asks how much money it can take and how fast its edge
dies.

> **A note on this file's history.** The placeholder that stood here before P3 began asked *"how
> much capital can this take before its own market impact eats the edge?"* and recorded, in advance,
> that daily bars might not be able to answer it. They cannot. That was measured rather than
> assumed, and the measurement became one of the benchmark's principal findings.

---

## What it measures, and what it refuses to

FlowState reports **constraint-based deployment capacity**: the largest AUM at which a strategy can
place every one of its trades without exceeding 1% of a session's traded value. This is arithmetic
on observed turnover and assumes nothing.

It does **not** report impact-erosion capacity — the AUM at which a strategy's own trading moves
prices enough to erase its edge. That needs a calibrated market-impact function, and §4 of
[RESULTS.md](RESULTS.md) is the measurement showing daily bars cannot supply one. **Every capacity
figure this benchmark produces is a constraint figure and none is an impact-erosion figure.** The
two answer different questions and only one is supported by the data.

## The three headline results, all negative

1. **Deployment capacity is small, and the least liquid position decides it.** Equal-weighted books
   trade every name in the same rupee size, so the participation limit binds on the thinnest holding
   first — a strategy of a hundred large caps can still be capacity-constrained by one of them.
2. **No factor's edge decays measurably within a quarter** — and none is statistically established
   at the shortest horizon, so the precondition for a decay study fails.
3. **Daily OHLCV cannot identify a transient-impact coefficient.** The obstruction is
   non-identification, not imprecision: two equally defensible weighting schemes disagree about the
   *sign*. Amihud illiquidity, computed on the same data as a control, is stable across a sample
   split — so the failure is in the model, not the data.

Plus a null on the novel contribution: **deployable size does not collapse when foreign
participation reverses.** That null rests on a derivatives-based flow proxy and cannot distinguish a
stable capacity from an uninformative proxy. Stated in the results, not the discussion.

## Contents

| File | What it is |
|---|---|
| [RESULTS.md](RESULTS.md) | Every number, generated. **Do not edit** — edit the template and regenerate |
| `RESULTS.template.md` | The source of the above |
| `paper_numbers.json` | Every published figure, each with the artifact it came from |
| [CORRECTIONS.md](CORRECTIONS.md) | Six defects found and repaired, with what each changed |

The paper is [`papers/flowstate.tex`](../../papers/flowstate.tex). **No number is typed into it.**
Every figure is a macro generated from an artifact, and `scripts/check_paper_numbers.py` fails the
build if a bare numeral appears in a claim position.

## Scope limitations a reader should know before the results

- **Value and quality factors are absent.** Both need company fundamentals this repository does not
  hold and could not buy at a zero data budget. The zoo is price-and-volume only, which weakens any
  comparison against published factor results.
- **`liquidity_size_proxy` is not a size factor.** It is built from trailing traded value, not
  market capitalisation, which needs a share count we do not have. It is named for what it is.
- **The flow series is derivatives, not cash.** No free source serves historical daily FII/DII
  cash-market net flow; NSE's endpoint returns the current session regardless of the date requested.
- **Returns are gross of costs.** Indian transaction costs are large enough to reverse a small edge.
- **Capacity is an extreme-value statistic.** The headline is a minimum over roughly twelve hundred
  sessions and is sensitive to any one of them. Percentile and drop-the-tightest-name variants are
  reported beside it.
- **One market, one window.** 2020–2024 contains one pandemic crash and one recovery.

## The process this benchmark changed

FlowState was validated on five standard factor strategies and was about to be frozen and written
up. Running it first over the machine-generated reference corpus exposed two defects the validation
set could not reach, and the corrections moved every published capacity figure **down** by between
4.3× and 26×.

That established a standing rule for every benchmark this lab builds:

> **build → validate on standard strategies → apply to the reference corpus → freeze → write up**

A benchmark's soundness cannot be established on the population that motivated it. Defects
concentrate in the tails, and a validation set of familiar, well-behaved strategies has no tails.

## Reproduction

```
python scripts/download_bhavcopy.py
python scripts/build_universe.py
python scripts/fetch_participant_flows.py
python scripts/diagnose_impact_identifiability.py
python scripts/build_flowstate.py
python scripts/run_corpus_capacity.py
python scripts/build_flowstate_numbers.py
python scripts/check_paper_numbers.py
```

Global seed 42. Every script writes a run manifest under `runs/` recording git SHA, config hash,
package versions and input file hashes. Development window 2020-01-01 to 2024-12-31; **the holdout
was never opened by any part of this project.**

## Status

**Frozen at v1.** Changes are versioned entries in [CORRECTIONS.md](CORRECTIONS.md), not edits.
The benchmark is now a fixed instrument, which is what makes it usable for comparing strategy
generators against one another.
