"""Point-in-time backtest engine and leakage-resistant cross-validation.

The engine's job is not to be fast. It is to make lookahead bias *hard to express*: a strategy
that tries to use information before it existed should raise, not quietly produce a better result.
"""
