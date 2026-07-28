# LOCKED measurement set — do not tune against this directory

**These cases are scored exactly once. Nothing in `src/audit/` may be changed in response to what
they show.**

## Why this directory exists separately from `refine/`

`tests/fixtures/refine/` is **burned**. The static auditor was tuned against it — its detectors were
rewritten after seeing which of those cases they missed — so the score there is training accuracy
and is not a result. Quoting it as the auditor's accuracy would be the exact error this repository
exists to detect.

This directory is the out-of-sample measurement. The number it produces is the one that goes in the
paper.

## Rules

1. **The auditor was frozen before a single case here was written.** Freeze commit: `71f8fd9`.
2. **No case here may inform a change to a detector.** If scoring reveals a miss, the miss is
   reported as a miss. It is not fixed and re-scored — that would convert this set into a second
   burned set and leave nothing honest to measure against.
3. **Scored once.** The command and its output are recorded in the checkpoint report.
4. **No case was copied or adapted from `refine/`.** See the independence note below.

## How independence was obtained

The cases were written by four separate processes that were each blocked from reading
`tests/fixtures/` entirely — including the refine set, the original leaky fixtures and the clean
fixtures — and blocked from reading `src/audit/`. They saw only `src/backtest/strategy.py`, the
interface every strategy implements.

That matters in both directions. A writer who had seen the refine cases would reproduce their
structure, and the score would measure recall on paraphrases rather than on new code. A writer who
had seen the detectors could write cases that trigger or evade them at will, in which case the score
would measure nothing at all.

## Difference from the refine set, deliberately introduced

The refine cases carry `# THE CHEAT:` comments marking the defect. Convenient while building, but a
generated strategy never announces its own leakage, and a benchmark whose defects are labelled in
the source is easier than the task it stands for.

**Nothing here marks its defect.** Each file reads as an honest strategy written by someone who did
not notice the problem. Ground truth lives outside the code, in the `labels_*.md` files in this
directory and in each filename's category prefix.

## Layout

```
locked/
├── leaky/          18 cases — 9 categories x 2, filename prefix gives the category
├── honest/         12 cases — 6 naming traps + 6 ordinary strategies
└── labels_*.md     ground truth: what the defect is and where, or why the case is clean
```

The nine leaky categories: `future_indexing`, `survivorship_selection`, `full_sample_fit`,
`full_sample_statistic`, `boundary_crossing_window`, `target_in_features`, `point_in_time_bypass`,
`future_dependent_ordering`, `snooped_parameter`.

One variant is deliberately excluded: a snooped parameter written as a bare literal with no visible
search. It is not statically detectable — the literal is byte-identical to a principled constant —
and it is recorded as a known blind spot in `benchmarks/alphaaudit/static_blind_spots.md` rather
than being counted as a miss.

## Scoring command

```
python scripts/attribution_report.py tests/fixtures/locked
```

Reports recall, per-class attribution, and the false-positive rate on the honest half. **Run once.**
