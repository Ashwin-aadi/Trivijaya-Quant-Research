"""Exceptions shared across the lab.

These are deliberately loud. In this project a silent wrong number is far more expensive than a
crash, so the backtester and data layer raise rather than warn-and-continue.
"""

from __future__ import annotations


class LabError(Exception):
    """Base class for every error this codebase raises on purpose."""


class PointInTimeError(LabError):
    """A piece of information was used before it was actually available.

    Raised by the backtest engine when an order's signal timestamp is not strictly earlier than
    its fill timestamp, and by the data layer when a feature would read a value stamped at or
    after the decision moment. This is the central correctness guarantee of the lab: lookahead
    bias must surface as an exception, never as a quietly better backtest.
    """


class DataIntegrityError(LabError):
    """Ingested data failed a sanity check (missing .meta.json, row count, hash mismatch, ...)."""


class ConfigError(LabError):
    """The configuration is missing a required value or holds an invalid one."""
