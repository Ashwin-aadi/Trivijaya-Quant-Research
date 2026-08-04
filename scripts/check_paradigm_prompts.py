"""Fail if any paradigm's prompt tells the model what the frozen evaluation stack looks for.

A scaffolded paradigm warned away from lookahead bias would score well on the static auditor by
construction, and its audit pass rate would measure how well the prompt describes the auditor
rather than how well the paradigm works. The failure would be invisible in the output and would
point in the direction that flatters the method, which is the worst combination available.

This is the machine version of the manual check in `reports/checkpoint_4.0.md` §2(a). It is not a
substitute for reading the prompts — a prompt can describe the auditor's criteria without using any
of these words — but it catches the careless version, and it catches it on every future edit.

Run: python scripts/check_paradigm_prompts.py
Exits 0 when clean, 1 when any prompt constant names the stack.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Vocabulary of the frozen stack. A paradigm prompt has no legitimate reason to contain any of it:
#: the task specification is P1's and deliberately says nothing about leakage, because warning a
#: generator away from the failure modes being measured would suppress the population under study.
BANNED: Final[tuple[str, ...]] = (
    "audit",
    "leak",
    "lookahead",
    "look-ahead",
    "point-in-time",
    "survivorship",
    "deflat",
    "overfit",
    "fragility",
    "capacity",
    "backtest overfitting",
)

MODULES: Final[tuple[str, ...]] = (
    "plain",
    "cot",
    "planning",
    "reflection",
    "multi_agent",
    "evolutionary",
)


def main() -> int:
    offences = 0
    checked = 0
    for name in MODULES:
        module = importlib.import_module(f"src.generate.paradigms.{name}")
        constants = {
            key: value
            for key, value in vars(module).items()
            if key.isupper() and isinstance(value, str)
        }
        checked += len(constants)
        for key, text in sorted(constants.items()):
            hits = [word for word in BANNED if word in text.lower()]
            if hits:
                offences += 1
                print(f"FAIL {name}.{key} names the frozen stack: {', '.join(hits)}")

    print(f"\n{offences} prompt constants name the frozen stack (checked {checked} across "
          f"{len(MODULES)} modules)")
    return 1 if offences else 0


if __name__ == "__main__":
    raise SystemExit(main())
