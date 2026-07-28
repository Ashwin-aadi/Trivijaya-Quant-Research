# `tests/fixtures/refine` — burned training data

This directory is **burned**. Its fixtures were written to be read, re-read and argued with while
the static auditor's rules were being developed, so the auditor has been shaped around them.

- Any precision, recall, or catch rate measured on this directory is **training accuracy**. It is
  a development diagnostic and is never reported as a result, in a checkpoint report, in
  `RESULTS.md`, or in a paper.
- Nothing here may be copied, adapted, or promoted into the locked evaluation sets in
  `tests/fixtures/leaky/` and `tests/fixtures/clean/`. Those measure the auditor on code it has
  not been tuned against, and reusing a fixture from here would destroy that property silently.
- Adding a fixture here is cheap and encouraged. Adding one to the locked sets is a decision for
  the PI.
