"""Honest strategies chosen to provoke a leakage auditor that matches names, not structure.

Nothing in this directory cheats. Every fixture reads the market only through its ``MarketView``,
which the engine has already truncated to sessions strictly before the decision date, and every
one stamps its signal with the last visible session rather than the session being traded.

Eight fixtures are naming traps. Each uses one word a naive detector keys on — ``target_weight``,
``final_weights``, ``label``, ``data``, ``_today``, ``all_symbols``, ``full_window``,
``latest_close`` — in the place where that word is the ordinary English name for what the code is
doing. The remaining ten use a construction that superficially resembles a leak and is correct: a
trailing mean, a positive one-session lag, a cross-section standardised within a single visible
date, a constructor that takes settings rather than a panel.

Any finding raised against this directory is a false positive. That is the point: it makes the
cost of a name-matching heuristic measurable instead of arguable.
"""
