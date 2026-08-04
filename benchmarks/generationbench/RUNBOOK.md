# GenerationBench — generation runbook

**For the PI, running Phase 4.1 alone.** Every command is copy-pasteable into Windows Terminal from
`C:\Trivijaya-Quant`. Nothing here needs me present.

**What you are doing:** generating 120 strategies from each of six methods, one method at a time,
about 17 hours of GPU in total. **What you are not doing:** evaluating any of them. Generation ends
at Checkpoint 4.1 and the audit, stress and capacity stages are Phase 4.2, which halts for approval
first.

---

## 0. Before you start — five minutes, once

```
cd C:\Trivijaya-Quant
.venv\Scripts\python.exe -m pytest tests/generate -q
.venv\Scripts\python.exe scripts/check_paradigm_prompts.py
ollama list
```

Expected: `52 passed`; `0 prompt constants name the frozen stack (checked 39 across 8 modules)`; and
`qwen2.5:7b-instruct-q4_K_M` present in the model list.

**If Ollama is not running**, start it and leave it running:

```
ollama serve
```

**If any test fails, stop and do not generate.** A failing harness produces a corpus that looks fine
and is not.

---

## 1. The timing probe — run this first, always

CLAUDE.md requires the run to be projected before it is committed to. This draws two strategies from
each arm and reports what the full run will cost.

```
.venv\Scripts\python.exe scripts/probe_paradigms.py --draws 2
```

Takes roughly 15–25 minutes, most of it G7. It writes to `runs\probe_<timestamp>\probe.json` and
**does not touch the corpus or any trial ledger**.

**What to do with the number it prints.**

| Projected total | What it means | What to do |
|---|---|---|
| under ~20 h | in line with the estimate | proceed to step 2 |
| 20–30 h | worse than estimated but survivable | proceed, and tell me the figure when I am back |
| over ~35 h | the estimate was badly wrong | **stop.** Run `--arm G7` alone to confirm it is the search arm, then reduce n. Do not start a run that cannot finish. |

The pre-registered estimate is **16.6 h structure-aware, 24.7 h as a call-count ceiling** — see
`PREREGISTRATION.md` Amendment 1.1.

---

## 2. Generate the arms, one at a time

**Run them in this order.** G1 first because it is the cheapest and the compute-matched control
cannot be sized without it; G7 last because it is two thirds of the run.

```
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G1 --n 120
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G2 --n 120
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G4 --n 120
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G5 --n 120
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G6 --n 120
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G7 --n 120
```

| Arm | Method | Rough time | Notes |
|---|---|---|---|
| G1 | plain prompting | ~40 min | also calibrates the control |
| G2 | chain of thought | ~1 h 15 | |
| G4 | planning | ~2 h | |
| G5 | reflection | ~1 h 40 | |
| G6 | graph of thoughts | ~2 h | |
| G7 | Monte Carlo tree search | ~9 h 30 | also uses ~4 h of CPU for backtests, inside that window |

**You can stop at any time with Ctrl-C.** Every completed draw is already on disk. Rerunning the
exact same command resumes from where it stopped — it skips indices that already have a
`draw_NNNN.json` and picks up the next one. Because seeding is deterministic, a resumed run produces
the identical corpus a single uninterrupted run would have.

**To do a fixed amount in one sitting** — useful for G7, which you will not want to do in one go:

```
.venv\Scripts\python.exe scripts/run_paradigm.py --arm G7 --n 120 --limit 20
```

That draws 20 and stops. Run it six times over several days and G7 is done.

---

## 3. Watch progress from a second terminal

Safe to run at any time, including mid-generation. It opens nothing for writing.

```
.venv\Scripts\python.exe scripts/paradigm_status.py
```

```
arm   paradigm                     draws  usable   yield    out tok  sec/draw  est left
----------------------------------------------------------------------------------------
G1    G1_plain                   120/120      19   15.8%    142,331      19.4        0m
G2    G2_cot                      47/120       8   17.0%     91,204      37.1       45m
G4    G4_planning                  0/120       -       -          -         -  not started
```

**What a healthy run looks like.** Yield somewhere between roughly 10% and 40%, seconds-per-draw
stable, output tokens growing steadily.

**What wrong looks like — check these, they are the point of the table:**

| Symptom | What it probably means |
|---|---|
| **yield 0% after 20+ draws** | the model is producing nothing conforming. Stop and tell me. |
| **yield above ~60%** | too good. Something is accepting code it should reject. Stop and tell me. |
| **out tok = 0 or blank** | Ollama is not returning token counts, and without them RULE 11's compute matching is fictional. Stop. |
| **sec/draw climbing steadily** | thermal throttling or a memory leak. Pause, restart, resume. |
| **G7 sec/draw under ~200** | it is not running its backtests. Stop and tell me. |

---

## 4. Verify each arm as it finishes

Run this after every arm, not only at the end. A defect found after G7 costs nine hours.

```
.venv\Scripts\python.exe scripts/verify_corpus.py --arm G1
```

Expected on a clean arm:

```
G1    120/120 draws    19 usable    134 candidates evaluated    134 ledger entries
all checks passed
```

It exits 0 when clean and 1 when not, and it prints exactly what failed. It checks that no draw
index is missing or duplicated, that every draw answered the theme its index assigns, that the task
prompt still hashes to the frozen P1 digest, that the model tag never changed, that every draw
recorded a non-zero token count, that usable draws have a file on disk matching the recorded source,
that unusable draws have no file, and that the hash-chained trial ledger verifies and holds exactly
as many entries as the draws claim attempts.

**A failure here is not something to work around.** Write down what it printed and stop that arm.

At the end, check everything at once:

```
.venv\Scripts\python.exe scripts/verify_corpus.py
```

---

## 5. When all six are done — STOP

```
.venv\Scripts\python.exe scripts/paradigm_status.py
.venv\Scripts\python.exe scripts/verify_corpus.py
```

Then stop. **Checkpoint 4.1 is a hard halt under RULE 1**, and it asks four questions that need your
judgment before anything is evaluated — including whether any arm shows the mode collapse H3
predicts, and whether any arm should be dropped. Dropping an arm at 4.1, on yield and redundancy
alone, is honest. Dropping one after its audit results are known is not, and there is no way to undo
having seen them.

Do not run `run_corpus_audit.py`, `run_corpus_backtest.py`, `run_stress_*`, or anything under
`src/audit`, `src/stress` or `src/capacity` against this corpus. That is Phase 4.2.

**Two things worth doing while you wait**, both of which Checkpoint 4.1 asks for:

- Read one strategy from the highest-yield arm and one from the lowest:
  `benchmarks\generationbench\corpus\G1\candidate_000.py` and the equivalent under `G7`. Are they
  qualitatively different, or is the scaffolding producing the same thing more slowly?
- Look at the duplicate rates in the status table yourself.

---

## 6. If something goes wrong

**Ollama stopped responding.** Ctrl-C, restart `ollama serve`, rerun the same command. It resumes.

**The machine rebooted.** Rerun the same command. It resumes.

**You want to start an arm over from scratch.** Delete that arm's directory —
`benchmarks\generationbench\corpus\G4` — and rerun. This deletes its trial ledger too, which is
correct: those trials belonged to a corpus that no longer exists. **Never delete individual draws
while keeping the ledger**, which would leave the ledger claiming a search that did not happen and
every Deflated Sharpe Ratio downstream too generous.

**`verify_corpus.py` says the prompt digest changed.** Something edited `src/generate/prompts.py`.
That makes P4 incomparable to P1's corpus, which is the compute-matched control. Stop and tell me;
do not regenerate around it.

**Anything under `src/audit`, `src/stress` or `src/capacity` shows as modified in `git status`.**
Stop immediately. Under RULE 7's amendment that invalidates both this study and the holdout. Do not
commit it; tell me what changed.

---

## What each file on disk is

```
benchmarks/generationbench/corpus/<arm>/
├── draw_0000.json        one per draw: outcome, theme, tokens, calls, candidates, timings, trace
├── candidate_000.py      the strategy source, written only for usable draws
├── trial_ledger.jsonl    hash-chained, append-only, one entry per candidate evaluated
└── summary.json          rewritten at the end of every sitting
runs/<timestamp>/manifest.json    git SHA, package versions, seed, config hash, model tag
```

The `draw_*.json` files are the corpus of record. The `.py` files are derived from them and are
checked against them by `verify_corpus.py`.
