"""Burned training material for hardening the static leakage auditor.

Nothing in this package is evidence. The fixtures here exist to be looked at repeatedly while the
auditor's rules are written and rewritten, which means any score measured against them is training
accuracy and carries no information about how the auditor behaves on code it has not shaped around.

The locked evaluation fixtures live in ``tests/fixtures/leaky`` and ``tests/fixtures/clean``. They
are not touched during rule development, and nothing from this package may be copied into them.
"""
