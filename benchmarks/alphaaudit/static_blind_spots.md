# Known blind spots of the static auditor

Defects `A_static` cannot find, recorded so the paper states them rather than leaving a reader to
assume the layer is complete. Each entry says what is missed, why it is missed, and what it would
take to catch — including the cases where the honest answer is that catching it would do more harm
than the miss does.

This file is maintained alongside the detector set. **Anything listed here belongs in the paper's
Limitations section.**

---

## 1. A snooped parameter with no visible search

**Missed:** a threshold, lookback or cutoff chosen by inspecting the whole sample and then written
into the source as a literal.

```python
# indistinguishable from an honest constant
self._threshold = 0.0273
```

**Why.** The defect leaves no structure. The literal is byte-identical whether it was derived from
a principled argument, taken from a published paper, or read off the value that maximised the
in-sample Sharpe. Nothing in the abstract syntax tree distinguishes the three cases, because the
distinguishing fact — how the number was arrived at — happened outside the file.

A *visible* search is a different matter and is caught: a loop or comprehension that scores
candidate values against data and keeps the best emits `SNOOPED_PARAMETER`, because the
optimisation is present in the code. Only the case where the search happened in the researcher's
head, or in a notebook that was not committed, is invisible.

**Why no detector was written.** The only remaining signal would be a confessional comment, and
keying on comments is the same string matching this layer removed everywhere else. A detector
flagging bare numeric literals would reject essentially every honest strategy in existence — a
lookback of 63, a cutoff of 0.8, a holding count of 10 are all literals — and the false-positive
cost would exceed anything the rule recovered. **PI decision, 2026-07-28: leave it uncaught and
report it.** An honestly-reported blind spot beats a brittle detector.

**What would catch it.** Not static analysis. Detecting parameter snooping needs either the trial
counter and Deflated Sharpe from `A_stat` — which is precisely why that layer exists and why the
ablation keeps the layers separate — or a record of the research process outside the artefact under
audit.

**Corpus effect.** Any reported class breakdown understates `snooped_parameter` by an unknown
amount, and a generator asked for strategies with hard-coded parameters would produce cases this
layer passes. The measured recall is therefore an upper bound with respect to this class.

---

## 2. The label among the features

**Missed:** the class attribution, not the rejection. `TARGET_IN_FEATURES` is not attributed to any
case where the leak is that a model's feature list includes the column it is predicting.

**Why.** Identifying which column is the label is semantic. Nothing structural separates
`target_return` from any other float column; the fact that it is the thing being predicted lives in
the researcher's intent. The previous detector matched the words `target`, `label` and
`future_return`, which is what rejected an honest strategy for naming a local `target_weight` — and
that rule was deleted rather than widened.

These cases are still **rejected**, because a strategy that receives a training frame at
construction commits `POINT_IN_TIME_BYPASS` and is caught there. Only the reason is wrong.

**Open taxonomy question.** As implemented, `TARGET_IN_FEATURES` fires on a forward-derived value
reaching the output, which is a sub-case of `FUTURE_INDEXING` rather than an independent class.
Whether to report it as a permanently unattributable class or merge it into `future_indexing` is a
decision recorded as outstanding in `DECISIONS.md`.

---

## 3. Data smuggled through module-level mutable state

**Missed:** the class attribution. A frame stored in a module-level container that is populated by a
function call, then read through a closure from inside `generate`.

```python
_REFERENCE: dict[str, pl.DataFrame] = {}

def register_reference(name, rows):
    _REFERENCE[name] = rows
```

**Why.** Module-level data is a declared tainted source, but only the direct binding form
`name = <call returning data>` is recognised. A container that is empty at definition and filled
later carries no evidence at the point it is defined.

**Status.** Caught, mislabelled. Not fixed, because the fix would have been written against the one
case that exposed it — see the note on refinement discipline in `DECISIONS.md`.

---

## 4. Reading the fill session's own bar

**Missed:** the class attribution. A strategy that reads the benchmark or price row at
`view.as_of` — the session the engine fills — rather than the last session that had printed.

**Why.** Telling those apart requires knowing what `MarketView.as_of` means in this engine. That is
a fact about the API, not about the code being audited, and encoding it would make the detector
specific to one interface.

**Compensating control.** The backtest engine raises `PointInTimeError` when a signal's timestamp is
not strictly prior to its fill timestamp, so this class of defect is caught at execution even where
the static layer does not name it. Static analysis is not the only line of defence and should not
be reported as though it were.

---

## 5. Missing embargo in cross-validation folds

**Missed:** the class attribution. Fold construction where training resumes immediately after the
test block, with overlapping label windows and no embargo either side.

**Why.** The defect is in arithmetic — `range(0, start) + range(stop, n)` against a label horizon —
and recognising it means interpreting the fold indices rather than the code structure.

**Compensating control.** `src/backtest/purged_cv.py` implements purging and embargo correctly, and
the engine's own tests assert no training index falls within the embargo window of any test index.
A strategy that builds its own folds bypasses that, which is the case here.

---

## A standing risk, recorded from experience

Two detectors in this module were described in an earlier checkpoint report as structural and were
not: `_BULK_DATA_PARAMS` and `FIT_METHODS` were both frozen sets of identifiers, and both were
defeated by renaming. **Any `frozenset` of names in `src/audit/static.py` should be treated as a
candidate brittle detector until it has been tested against a renamed case.** The remaining
sets — `DATA_METHODS`, `AGGREGATIONS`, `WINDOW_METHODS`, `BACKWARD_FILL_METHODS` — name *library
methods* rather than user-chosen identifiers, which is a materially different thing, since a
strategy author does not get to rename `polars`. That distinction is the reason they are considered
sound, and it is the thing to re-check if one of them is ever defeated.
