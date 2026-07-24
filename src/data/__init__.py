"""Data layer: trading calendar, point-in-time universe, prices, and corporate actions.

Everything here is responsible for one thing above all else — never letting a future value reach a
past decision. Read the module docstrings before changing anything; the invariants are load-bearing.
"""
