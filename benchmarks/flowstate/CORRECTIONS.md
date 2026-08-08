# FlowState — corrections

Append-only. Every entry records a defect found in this benchmark, what it changed, and how it was
found. Entries before the v1 freeze are pre-freeze repairs; entries after it are versioned
corrections and require a version bump.

| # | Found | Defect | Effect on published figures |
|---|---|---|---|
| C1 | pre-freeze, Phase 3.3 | Floating-point residue counted as trading | One strategy's capacity reported as 4.5×10²³ crore |
| C2 | pre-freeze, Phase 3.3 | Median session capacity used as the headline | **All capacity figures overstated by 4.3× to 26×** |
| C3 | pre-freeze, Phase 3.1 | Universe expansion collapsed by an as-of join | 18 symbols instead of 184; every downstream number wrong |
| C4 | pre-freeze, Phase 3.1 | Monthly book formation | 59 observations, ragged curves, no t-statistic above 1.5 |
| C7 | **post-freeze**, 2026-08-08 | Two price-panel loaders disagree; released capacity used the narrower | Reproducibility defect; capacity insensitive in the tested sample, impact on P3 not yet measured |
| C5 | pre-freeze, Phase 3.0 | Per-symbol reversal test reported without a power bound | An underpowered proportion presented as evidence |
| C6 | pre-freeze, Phase 3.2 | The validation-gap magnitude was overstated in a checkpoint report | Claimed ~20×; the correct figure is 3.2× |

---

## C1 — floating-point residue counted as trading

A strategy holding 100 names at 0.01 each and never rebalancing produces weight changes of
**1.7×10⁻¹⁸** from recomputing an unchanged number. Dividing a participation limit by that residue
reported its deployable capacity as **4.5×10²³ crore**.

Measured scope: **150 of 156 corpus strategies had no such trades at all**; one was 99.7% residue and
four others between 0.2% and 8.7%. So this was one broken strategy, not a broken corpus — but the
broken value was in the artifact.

**Fix:** `constraints.min_traded_fraction: 1.0e-9` in `config.yaml`, applied in
`turnover_by_session`. **Found by:** running the machine-generated corpus through the benchmark
before freezing it. The five standard factor strategies rebalance substantially every session and
never exercised this path.

## C2 — the median session capacity was the wrong headline

**This is the largest correction in P3 and it superseded a result the PI had already approved.**

Capacity was reported as the *median* across sessions. That is the wrong summary when turnover is
uneven: a near-static book has near-zero rebalancing on almost every session, so the one tightly
constrained session that builds the position was drowned by eleven hundred that barely trade.

A subtlety worth recording, because it was diagnosed wrongly first. The initial report claimed the
formula "never counts acquiring the initial position." **That was false** — `turnover_by_session`
already treats a strategy's first session as a full acquisition from zero, verified directly. The
defect was that the *summary statistic discarded it*.

**Fix:** `binding_capacity_inr`, the minimum across every session traded, becomes the headline;
`entry_capacity_inr` is reported as its own fact; `median_capacity_inr` is retained only as a
typical-session statistic.

**Effect on the standard factor strategies:**

| Factor | Reported at Checkpoint 3.1 | Corrected | Overstatement |
|---|---|---|---|
| liquidity_size_proxy | ₹33.9 cr | ₹7.91 cr | 4.3× |
| illiquidity | ₹32.0 cr | ₹7.22 cr | 4.4× |
| momentum_12_1 | ₹31.4 cr | ₹6.94 cr | 4.5× |
| low_volatility | ₹32.9 cr | ₹6.78 cr | 4.8× |
| reversal_1m | ₹21.2 cr | ₹0.81 cr | 26.2× |

Entry is the binding session for only **32 of 156** corpus strategies, so the median was hiding more
than the entry constraint alone.

## C3 — the universe expansion collapsed

`daily_universe` used an as-of join directly against the universe frame. An as-of join returns **one
right row per left row**, so each session received a single arbitrary symbol instead of its whole
membership: **18 symbols instead of 184**. It produced a complete, plausible set of capacity and
decay numbers before anyone checked.

**Found by:** a logged row count that read 1,233 symbol-days where 122,464 were expected. Now
guarded by an assertion inside `daily_universe` and by a test.

## C4 — monthly book formation

Books were formed every 21 sessions on the reasoning that the signals barely move between sessions.
That reasoning confuses how smooth a *signal* is with how precisely its mean return is *estimated*.
It left 59 observations, every decay curve was visibly ragged, and no t-statistic reached 1.5.

**Fix:** daily formation, with overlap corrected in the standard error rather than avoided by
discarding data.

## C5 — an underpowered proportion presented as evidence

Checkpoint 3.0 reported *"48.0% of symbols have a negative reversal beta — indistinguishable from a
coin flip"* as evidence that no reversal exists. A single symbol's daily test can only detect a beta
of **0.368** or larger; the effects in play are of order 0.02–0.10. The test had essentially no
power, so the proportion was always going to land near half regardless of the truth.

**Fix:** `minimum_detectable_beta` now accompanies every reversal estimate. Pooled across symbols the
reversal *is* significantly negative, which contradicts the original wording — and the identification
argument that replaced it is stronger.

## C6 — the validation-gap magnitude was overstated

`reports/checkpoint_3.3.md` reported the validation set as spanning **1.6×** against a corpus
spanning **31×**, and the surrounding text described the validation population as roughly twenty
times narrower.

Those two spans were measured on **different statistics**: 1.6× was the span of *median* capacity,
which C2 then established was the wrong measure. Measured consistently on the corrected binding
capacity, the standard factor strategies span **9.8×** and the corpus spans **30.9×** — a ratio of
**3.2×**, not 20×.

The finding survives — the validation population is narrower than the application population, and
both defects lived in the gap — but it is a threefold difference, not a twentyfold one, and the
paper states the smaller figure.

## C7 — two price-panel loaders disagree, and the released capacity numbers used the narrower one

**Post-freeze. Found 2026-08-08, during P4 Phase 4.2. Scoped by PI ruling the same day.**

**The defect.** `data/processed/prices_adjusted.parquet` begins on **2019-06-25**, while trading
begins at `dev_start`, **2020-01-01**. Those 188,853 pre-window rows are the warm-up a lookback
strategy needs on its first trading session. Two loaders treat them differently:

| loader | range | symbols | used by |
|---|---|---|---|
| direct read, unfiltered | 2019-06-25 → 2024-12-31 | 2,697 | `run_corpus_backtest.py` — every published Sharpe |
| `load_dev_panel()` | 2020-01-01 → 2024-12-31 | 185 | `persist_positions.py`, `persist_real_returns.py`, `run_frontier_capacity.py` |

**FlowState's released capacity figures were computed from `data/interim/positions/`, which
`persist_positions.py` wrote through the narrower loader.** The Sharpe figures they sit beside came
from the wider one. A strategy with a 60-session lookback therefore has capacity measured on a book
that was empty for its first weeks and a Sharpe measured on a book that was not.

**How it was found.** Not by inspection — by a crash. A P4 candidate weighting the calmest 30% of
symbols holding a complete 60-session window divides by that count, which is zero in January 2020
without warm-up.

**Measured effect, n = 8 G1 strategies.**

- The **unfiltered panel reproduces the released Sharpe figures** to within **0.0012** in all eight.
- The **truncated panel does not.** Seven of eight produce books differing on 9 to 23 sessions, and
  on most of those the L1 distance between books is exactly 1.0 — the truncated run holds nothing at
  all while the other is fully invested. Six move in the third decimal; `candidate_029` moves from
  **−0.7652 to −2.3878**.
- **Capacity appeared insensitive** in the tested sample. Recomputing P4's G1 arm through the
  unfiltered panel left its capacity summary identical to two decimals — min 0.19, median 2.60, max
  23.65 crore — because binding capacity is a minimum over ~1,200 sessions and does not notice a cold
  start of twenty.

**What this entry does *not* claim.** **P3's published conclusions are not asserted to be invalid.**
The insensitivity above was measured on P4's G1 arm, not on P3's own corpus, which has not been
rerun. This is a versioned correction notice recording a reproducibility defect in the evaluation
infrastructure; the impact assessment on P3's figures is outstanding work, not a completed finding.

**Status.** P4's capacity is computed from the unfiltered panel (PREREGISTRATION.md Amendment 4
§4.1). No frozen module was modified. `load_dev_panel` is unchanged, so P3's artefacts remain exactly
reproducible as published, and this entry is the record of what they were produced with.
