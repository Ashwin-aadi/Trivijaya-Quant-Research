"""Formatting helpers for figures that are emitted into a paper rather than into a log.

A number that reaches a paper has to survive a different kind of scrutiny from one that reaches a
log line: it is quoted, compared against a table three pages away, and read by someone deciding
whether to believe the rest. These helpers exist so that every such number is rendered by the same
code, in one place, and so that rounding is a property of the pipeline rather than of whoever was
typing at the time.

Nothing here computes anything. It only decides how a computed value is written down.
"""

from __future__ import annotations

import math
import re

#: LaTeX macro names may contain letters only. Anything a caller passes is checked against this
#: rather than silently mangled, because a macro whose name failed to typeset would produce a paper
#: with a visible ``\rsNPrimary1`` in the body, and that is exactly the failure this module exists
#: to make impossible.
_MACRO_NAME = re.compile(r"\A[A-Za-z]+\Z")


def macro_name(name: str) -> str:
    """Validate a LaTeX macro name, raising rather than emitting something that will not compile."""
    if not _MACRO_NAME.match(name):
        raise ValueError(f"macro name must be letters only, got {name!r}")
    return name


def fixed(value: float, places: int = 3) -> str:
    """A value to a fixed number of decimals, with a real minus sign rather than a hyphen."""
    if not math.isfinite(value):
        raise ValueError(f"refusing to typeset a non-finite value: {value}")
    return f"{value:.{places}f}".replace("-", "$-$")


def signed(value: float, places: int = 3) -> str:
    """As :func:`fixed`, but always carrying its sign.

    Used for scores that can fall either side of zero — an out-of-sample R-squared of ``0.024``
    and one of ``-0.024`` are different findings, and a reader skimming a column of them should
    not have to look twice.
    """
    if not math.isfinite(value):
        raise ValueError(f"refusing to typeset a non-finite value: {value}")
    return f"{value:+.{places}f}".replace("-", "$-$").replace("+", "$+$")


def integer(value: int | float) -> str:
    """An integer with LaTeX-safe thousands separators: ``1233`` becomes ``1{,}233``."""
    whole = int(value)
    if whole != value:
        raise ValueError(f"refusing to typeset {value} as an integer")
    return f"{whole:,}".replace(",", "{,}")


def percent(fraction: float, places: int = 1) -> str:
    """A fraction rendered as a percentage, without the sign, which the caller supplies."""
    return fixed(100.0 * fraction, places)


def plain(rendered: str) -> str:
    """Strip the LaTeX rendering back to something a Markdown reader sees correctly.

    The macro table is written once and consumed twice — by the paper, which wants ``$-$`` and
    ``1{,}233``, and by the benchmark's ``RESULTS.md``, which wants ``-`` and ``1,233``. Deriving
    the second from the first is the only arrangement in which the two cannot disagree.
    """
    out = rendered.replace("$-$", "-").replace("$+$", "+").replace("{,}", ",")
    return out.replace(r"\times10^{", "e").replace("}", "").replace(r"\_", "_")


def scientific(value: float, places: int = 1) -> str:
    """A value in LaTeX scientific notation, for quantities too small to read as decimals."""
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.{places}f}\times10^{{{exponent}}}"
