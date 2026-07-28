# `tests/fixtures/refine` — burned training data

This directory is **burned**. Its fixtures were written to be read, re-read and argued with while
the static auditor's rules were being developed, so the auditor has been shaped around them.

- Any precision, recall, or catch rate measured on this directory is **training accuracy**. It is
  a development diagnostic and is never reported as a result, in a checkpoint report, in
  `RESULTS.md`, or in a paper.
- Nothing here may be copied, adapted, or promoted into the locked evaluation set in
  `tests/fixtures/locked/`. That set measures the auditor on code it has not been tuned against,
  and reusing a fixture from here would destroy that property silently.
- Adding a fixture here is cheap and encouraged. Adding one to the locked set is a decision for
  the PI.

**Correction to an earlier version of this file.** It named `tests/fixtures/leaky/` and
`tests/fixtures/clean/` as the locked evaluation sets. They are not, and had not been for some
time. The three leaky fixtures are the positive controls the detectors were originally built
against, and both directories were re-run as regression checks after every rule change during
refinement — which makes them burned in exactly the same way this directory is. Treating their
zero-false-positive rate as an out-of-sample result would have overstated the auditor. The genuine
out-of-sample measurement is `tests/fixtures/locked/`, written after the auditor was frozen.
