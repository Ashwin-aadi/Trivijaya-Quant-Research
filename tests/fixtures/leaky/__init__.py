"""Strategies that cheat on purpose.

These are permanent assets, not throwaway tests. Each one commits a specific, well-known form of
lookahead bias, and each must produce an obviously absurd result. They are the positive controls
for the leakage auditor: a detector that misses any of these does not work.

Each cheat is deliberately the *only* thing wrong with its strategy, so that when the auditor
flags it the reason is unambiguous.
"""
