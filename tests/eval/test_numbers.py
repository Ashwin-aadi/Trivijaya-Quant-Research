"""The paper's figures pass through these functions, so a defect here misprints a published number.

Every test below fixes an expected string rather than a property, because the failure mode that
matters is not "the function raised" but "the paper says 0.24 where the artifact says 0.238".
"""

from __future__ import annotations

import math

import pytest
from scripts.check_paper_numbers import ALLOWED, PAPER, _bare_numerals, _body

from src.eval.numbers import fixed, integer, macro_name, percent, plain, scientific, signed


def test_a_minus_sign_is_typeset_as_a_minus_not_a_hyphen() -> None:
    """A hyphen renders shorter than a minus and reads as a typo in a results table."""
    assert fixed(-1.742) == "$-$1.742"
    assert fixed(1.742) == "1.742"


def test_signed_always_carries_its_sign() -> None:
    # +0.024 and -0.024 are different findings; a reader scanning a column must not have to look
    # twice to tell which one a cell holds.
    assert signed(0.024) == "$+$0.024"
    assert signed(-0.036) == "$-$0.036"


def test_rounding_is_to_the_requested_places_and_does_not_drop_trailing_zeros() -> None:
    assert fixed(0.5, 3) == "0.500"
    assert signed(0.2, 1) == "$+$0.2"


def test_a_non_finite_value_raises_rather_than_printing_nan_into_the_paper() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            fixed(bad)
        with pytest.raises(ValueError):
            signed(bad)


def test_integers_are_grouped_in_a_way_latex_will_not_reflow() -> None:
    # A bare comma in LaTeX maths gets extra space after it; the braced form does not.
    assert integer(1233) == "1{,}233"
    assert integer(109) == "109"


def test_a_fractional_value_cannot_be_typeset_as_an_integer() -> None:
    """Guards a count becoming a mean through an upstream change, unnoticed."""
    with pytest.raises(ValueError):
        integer(108.5)
    assert integer(109.0) == "109"


def test_percent_and_scientific() -> None:
    assert percent(0.961) == "96.1"
    assert scientific(4.7379e-08) == r"4.7\times10^{-8}"
    assert scientific(0.0) == "0"


@pytest.mark.parametrize("name", ["rs1", "rs_name", "rsName2", "", "rs name"])
def test_macro_names_with_anything_but_letters_are_rejected(name: str) -> None:
    """LaTeX would silently mis-parse these, printing the offending characters into the body."""
    with pytest.raises(ValueError):
        macro_name(name)


def test_valid_macro_name_passes_through() -> None:
    assert macro_name("rsTierAgreementFragility") == "rsTierAgreementFragility"


def test_plain_strips_the_latex_rendering_for_markdown() -> None:
    """The macro table is written once and read twice; the Markdown view is derived."""
    assert plain(signed(-0.036)) == "-0.036"
    assert plain(signed(0.212)) == "+0.212"
    assert plain(integer(1233)) == "1,233"
    assert plain(scientific(4.7379e-08)) == "4.7e-8"
    assert plain(r"mean\_herfindahl") == "mean_herfindahl"


# --- the checker that keeps transcribed figures out of the paper -----------------------


def test_the_checker_catches_a_transcribed_figure() -> None:
    """The whole point of the gate: a number typed into a claim must not pass."""
    body = "The model reached an out-of-sample R-squared of 0.212 on the log target."
    found = _bare_numerals(body)
    assert [token for _, token, _ in found] == ["0.212"]


def test_the_checker_ignores_a_generated_figure() -> None:
    body = r"The model reached an out-of-sample $R^2$ of \rsRegimesLogForestRsquared\ on it."
    assert _bare_numerals(body) == []


def test_the_checker_ignores_structural_latex() -> None:
    """Column widths and row spacing carry digits and state nothing."""
    body = r"\begin{tabular}{@{}p{0.20\linewidth}rr@{}} a & b & c \\[0.45em]"
    assert _bare_numerals(body) == []


def test_every_allowed_literal_carries_a_written_reason() -> None:
    """A number admitted to the allowlist without a justification is an unaudited claim."""
    assert all(isinstance(why, str) and len(why) > 20 for why in ALLOWED.values())


# --- the negative control, run against the real paper -----------------------------------
#
# The PI ran this by hand before authorising the freeze: type a figure into a claim sentence and
# confirm the checker rejects it. A gate nobody has seen fail is indistinguishable from a gate that
# cannot fail, so the control is kept here rather than left as a one-time manual act.


def test_the_committed_paper_states_no_figure_of_its_own() -> None:
    assert _bare_numerals(_body(PAPER.read_text(encoding="utf-8"))) == []


def test_a_figure_typed_into_the_real_paper_is_rejected() -> None:
    """The other direction, and the one that matters: tampering must not pass."""
    body = _body(PAPER.read_text(encoding="utf-8"))
    # Substitute a macro in a claim sentence for the value it renders to, exactly as a careless
    # edit would. The paper must stop being clean the moment it states a number itself.
    tampered = body.replace(r"\rsShortcutSpearman", "0.963", 1)
    assert tampered != body, "the macro this control depends on is no longer used in the paper"
    assert "0.963" in [token for _, token, _ in _bare_numerals(tampered)]
