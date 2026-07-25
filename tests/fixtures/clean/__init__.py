"""Honest strategies used to measure the auditor's false-positive rate.

None of these cheat. They are simple, well-known rules, and most are expected to be unprofitable
after costs — that is fine and is not what they are for. What matters is that an auditor which
flags any of them as leaky is producing a false positive, so any accidental lookahead here would
corrupt the measurement these fixtures exist to provide.
"""
