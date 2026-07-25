"""The three-layer strategy auditor.

Each layer answers a different question and can be ablated independently:

* ``static``   — does the code read information it could not have had?
* ``stat``     — does the result survive an honest accounting of how many things were tried?
* ``semantic`` — does the stated reasoning match what the code actually does?

They are deliberately separate. A strategy can pass any one and fail another, and the point of the
project is to measure which layers earn their place rather than to assume all three do.
"""
