# AlphaAudit — corrections log

Every correction applied to this benchmark after its first freeze, with what changed, what did
not, and why. A benchmark that is quietly edited is not a benchmark, so nothing here is removed:
superseded artifacts are retained beside their replacements under `runs/pooled/` with a
`_v1.0_superseded` suffix.

| Version | Date | Nature |
|---|---|---|
| 1.0 | 2026-08-01 | First freeze. |
| **1.1** | **2026-08-01** | **Implementation bug in the abstention ranking. Semantic-inclusive configurations rerun.** |

---

## v1.1 — sign error in the auditor-confidence ranking

### What was wrong

`soundness()` in `scripts/run_ablation.py` converted each layer's stored `confidence` into a
ranking score by negating it:

```python
return -float(record.get("confidence", 0.0))     # v1.0
```

This assumed every layer stores *confidence that something is wrong*. Two of the three do:

* **static** stores a finding count — `0.0` when clean, `1.0`/`2.0`/`3.0` when flagged;
* **statistical** stores a constant `1.0`, on records it rejects without exception (631 of 631).

The **semantic** layer does not. It stores the model's confidence **in its own label**: about
`0.95` for a confident `consistent` and about `0.85` for a confident defect. Negating that ranked
the strategies the semantic layer had *rejected* above the ones it had *cleared* — the ranking was
inverted, so at low coverage the pipeline retained precisely the candidates the layer had
condemned.

Measured on the 225 rankable candidates: Spearman correlation between the v1.0 and v1.1 semantic
ranking is **−0.972**, and the retained set at 5% coverage went from **11 of 11 semantically
rejected** to **0 of 11**.

### Why no test caught it

Each layer's own tests check its labels, which were correct. `tests/eval/test_abstention.py`
checks that the curve is computed correctly from a ranking, which it was. The defect lived only in
the join between them — a ranking computed correctly from a quantity that meant the opposite of
what the calling code believed. `tests/eval/test_soundness_sign.py` now asserts that join
directly, per layer, in the direction a reader can check by eye.

### The fix

Sign comes from the layer's own `rejected` verdict rather than from an assumption about what
`confidence` means:

```python
confidence = float(record.get("confidence", 0.0))
return -confidence if record.get("rejected", True) else confidence     # v1.1
```

Unscored candidates still contribute `0.0`, which now sits between a confident acceptance and a
confident rejection.

### What was rerun, and what was not

The correction is a general rule, but on this corpus it is **provably confined to the semantic
layer**, because the static layer records `rejected` exactly when its finding count is non-zero and
the statistical layer rejects every record it scores. Both therefore produce values identical to
v1.0 under the new rule. This is a property of the frozen data, not an argument, and
`tests/eval/test_soundness_sign.py::test_the_correction_leaves_the_other_layers_bit_identical`
checks it against the data.

The three non-semantic configurations were recomputed anyway and came back **bit-identical**
(∆AUAP = +0.0000 in every case), which is a stronger demonstration than skipping them.

Nothing else was touched: no methodology, thresholds, prompts, model outputs, backtests, universe,
cost model, trial counter, or holdout contents. The corpus, the audit labels and the two
performance files are byte-for-byte the ones v1.0 used. The only difference between the runs is the
two lines above.

### Holdout AUAP — v1.0 vs v1.1

Population n = 225, fixed by development eligibility. Random-rejection 95% interval
**[−1.2208, −0.8600]**, unchanged (it does not depend on the ranking).

| Layers | AUAP v1.0 | **AUAP v1.1** | ∆ | P(0.05) v1.0 | **P(0.05) v1.1** | Position vs random |
|---|---|---|---|---|---|---|
| semantic | −1.1441 | **−1.1967** | −0.0526 | −0.5147 | **−1.1347** | inside |
| static | −1.2529 | **−1.2529** | +0.0000 | −1.1176 | **−1.1176** | below |
| static + semantic | −1.2249 | **−1.2782** | −0.0533 | −0.8660 | **−1.1805** | below |
| statistical | −1.3497 | **−1.3497** | +0.0000 | −3.0643 | **−3.0643** | below |
| semantic + statistical | −1.2344 | **−1.3777** | −0.1433 | −0.5147 | **−3.2931** | below |
| static + semantic + statistical | −1.2849 | **−1.4184** | −0.1336 | −0.8660 | **−3.2931** | below |
| static + statistical | −1.4339 | **−1.4339** | +0.0000 | −3.2931 | **−3.2931** | below |

### Development AUAP — v1.0 vs v1.1 (diagnostic only)

Random band **[−0.3706, −0.0270]**.

| Layers | AUAP v1.0 | **AUAP v1.1** | ∆ | P(0.05) v1.0 | **P(0.05) v1.1** |
|---|---|---|---|---|---|
| semantic | −0.2712 | **−0.2917** | −0.0205 | −0.1088 | **+0.2910** |
| static + semantic | −0.3055 | **−0.3310** | −0.0255 | −0.0918 | **+0.3142** |
| static | −0.3327 | **−0.3327** | +0.0000 | +0.2073 | **+0.2073** |
| statistical | −0.3995 | **−0.3995** | +0.0000 | −1.3599 | **−1.3599** |
| semantic + statistical | −0.3290 | **−0.4124** | −0.0834 | −0.1088 | **−1.3494** |
| static + semantic + statistical | −0.3514 | **−0.4374** | −0.0860 | −0.0918 | **−1.3494** |
| static + statistical | −0.4550 | **−0.4550** | +0.0000 | −1.3494 | **−1.3494** |

### Effect on the conclusions

**The headline conclusion is unchanged: the null is not rejected.** No configuration beat random
rejection under v1.0 and none does under v1.1. The correction moved every affected number in the
direction that *strengthens* that finding, not away from it:

* The best holdout AUAP is now **−1.1967 (`semantic` alone)**, previously −1.1441, and it is still
  inside the random interval. The identity of the best configuration did not change.
* Most curves still **fall** as coverage tightens; with the ranking corrected they fall harder.

### A second error, found while checking the first

v1.0's prose stated that "five configurations sit inside the random interval; two sit below it",
and its table annotated only `static` and `static + statistical` as below. **That was a
miscount, present in v1.0 and unrelated to the ranking bug.** Six of the seven v1.0 AUAPs were
already below the interval's lower bound of −1.2208; the annotation was internally inconsistent,
marking `static` at −1.2529 as below while leaving `statistical` at −1.3497 unmarked. The count is
therefore **six of seven below the interval in both versions**, and only `semantic` alone inside
in both.

The correction to the ranking did not change that count. It is stated separately here so that
nothing in the v1.1 write-up claims credit for fixing an error the v1.1 ranking did not cause.
The v1.0 numbers themselves were correct; only the sentence counting them was wrong.

### Two consequences of the ranking correction, stated plainly

Neither is left for a reader to notice.

**The corrected ranking performs worse out of sample than the inverted one.** Every affected AUAP
moved down. It would be an overclaim to read that as the semantic layer being anti-informative:
both orderings sit inside or below a random interval that is 0.36 wide, and a difference of 0.05
between two null results is not a finding. The honest statement is that the layer carries no usable
out-of-sample ordering information in either direction.

**In sample, the correction does what a working auditor should do, and it does not survive.**
Development `P(0.05)` for `semantic` rose from −0.1088 to **+0.2910**, and for `static + semantic`
from −0.0918 to **+0.3142** — a correctly-oriented semantic layer visibly concentrates the better
strategies at tight coverage on the data it was developed against. The same ranking on the holdout
gives −1.1347 and −1.1805. That gap between an in-sample selectivity that looks real and an
out-of-sample one that does not is the paper's subject, arrived at here by accident.

### One further observation, not corrected

Under both v1.0 and v1.1 the **statistical** layer contributes no ordering information among the
candidates it scores: it rejects all 631 at confidence 1.0, so every scored candidate ties at
−1.0 and the layer's only discrimination is between scored and unscored. That is a genuine
weakness of the layer as implemented, but it is not an inversion and it was not in the scope
authorised for this correction. It is recorded here and left for the PI to rule on.

### Reproduction

```bash
python scripts/run_ablation.py --corpus runs/pooled/candidates
python scripts/run_ablation.py --corpus runs/pooled/candidates --holdout --authorised-by "<who, when>"
python scripts/plot_abstention.py --ablation runs/pooled/ablation_holdout.json
pytest tests/eval/test_soundness_sign.py
```

The holdout authorisation recorded in `runs/pooled/ablation_holdout.json` is the PI's of
2026-08-01. Under Rule 7 the holdout is otherwise closed; this evaluation was authorised
explicitly as the repair of an erroneous evaluation pipeline, and no tuning of any kind followed
it.
