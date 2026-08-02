"""Fail the build if the RegimeStress paper states a figure that did not come from an artifact.

The PI's freeze precondition was that every quantitative claim in the paper be generated from the
final artifacts rather than copied from an intermediate report. ``build_paper_numbers.py`` makes
that possible; this script makes it enforceable, because a generator nobody checks against is a
convention rather than a guarantee.

Three checks, all hard failures:

1. **Every macro the paper uses is defined.** An undefined macro typesets as nothing or as an
   error, so this catches a claim whose supporting number was removed from the pipeline.
2. **Every macro the pipeline defines is used somewhere.** An unused macro is a claim that was
   deleted from the paper but is still being computed, or a renamed one — either way, drift.
3. **No bare numeral survives in a claim position.** The body is stripped of structural LaTeX
   (column specifications, box geometry, float placement, the bibliography) and every remaining
   numeric literal must appear in :data:`ALLOWED` with a written reason. A number that is genuinely
   a result cannot be added to that list honestly.

Usage:
    python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

from src.common.log import configure_logging, get_logger  # noqa: E402

_log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "papers" / "regimestress.tex"
NUMBERS = ROOT / "papers" / "regimestress_numbers.tex"
TEMPLATE = ROOT / "benchmarks" / "regimestress" / "RESULTS.template.md"

#: Numeric literals allowed to appear in the paper's prose, each with the reason it is not a
#: result. Anything that is a measurement belongs in a macro; nothing measured can be justified
#: here, which is what makes the list safe to keep short.
ALLOWED = {
    "2015": "the start of the market history described in the introduction, not a measurement",
    "2020": "the evaluation window's first year, stated as context",
    "2022": "a year named in an example about drifting regime labels, not a measurement",
    "2024": "the evaluation window's last year, stated as context",
    "95": "the nominal level of a confidence interval, a choice rather than a result",
    "95th": "the same level, written as an ordinal",
    "1": "reference to a single item, and the constant in log(1+x)",
    "2": "the exponent in R^2, and the two-session decode lag",
    "5": "the number of rows removed by the drop-5 trimming, a protocol constant",
    "12": "the exponent in the 1e-12 identical-series tolerance, a protocol constant",
    "15": "the exponent in the 1e-15 knife-edge perturbation, a protocol constant",
    "42": "the global random seed, fixed by the project charter",
    "7": "the parameter count of the local model, in billions, as a model identifier",
    "0.5": "the reporting threshold for a large Sharpe swing, a presentation choice",
    "1.000": "an exact correlation of one between a duplicated pair; the defect, not a measurement",
    "0": "the temperature of the local model, a configuration constant",
}

#: Structural LaTeX that legitimately contains digits and never states a claim. Stripped before the
#: numeral scan so the check does not drown in column widths.
_STRUCTURAL = (
    re.compile(r"%.*$", re.MULTILINE),                      # comments
    re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL),
    re.compile(r"\\begin\{tabular\}\{[^}]*\}"),             # column specifications
    re.compile(r"p\{[0-9.]*\\linewidth\}"),                 # ...and their widths
    re.compile(r"\\(?:hspace|vspace|rule|multicolumn)\*?\{[^}]*\}(?:\{[^}]*\})?"),
    re.compile(r"\\begin\{(?:table|figure|tcolorbox|finding|selfreport)\}(?:\[[^\]]*\])?"),
    re.compile(r"\\(?:S)?ref\{[^}]*\}"),                    # cross-references
    re.compile(r"\\cite[a-z]*\{[^}]*\}"),                   # citation keys carry years
    re.compile(r"\\\\\[[^\]]*\]"),                          # inter-row spacing
    re.compile(r"[a-zA-Z]+\s*=\s*[0-9.]+(?:em|pt|cm|in|ex)?"),  # option assignments
    re.compile(r"\$10\^\{-?\d+\}\$"),                       # scientific notation in prose
    re.compile(r"\bT\d+(?= ---)"),                          # threats-to-validity item labels
    re.compile(r"\\rs[A-Za-z]+"),                           # the generated macros themselves
    re.compile(r"\\[a-zA-Z]+"),                             # any remaining control sequence
)

_NUMERAL = re.compile(r"\d+(?:\.\d+)?(?:st|nd|rd|th)?")


def _body(text: str) -> str:
    """The paper after ``\\begin{document}``. The preamble is styling and states nothing."""
    _, _, after = text.partition(r"\begin{document}")
    if not after:
        raise ValueError("no \\begin{document} in the paper")
    return after


def _defined(text: str) -> set[str]:
    return set(re.findall(r"\\newcommand\{\\(rs[A-Za-z]+)\}", text))


def _used(text: str) -> set[str]:
    return set(re.findall(r"\\(rs[A-Za-z]+)", text))


def _bare_numerals(body: str) -> list[tuple[int, str, str]]:
    """Numeric literals surviving the structural strip, as ``(line number, token, line)``."""
    stripped = body
    for pattern in _STRUCTURAL:
        stripped = pattern.sub(" ", stripped)
    findings: list[tuple[int, str, str]] = []
    for number, line in enumerate(stripped.splitlines(), start=1):
        for token in _NUMERAL.findall(line):
            if token not in ALLOWED:
                findings.append((number, token, line.strip()))
    return findings


def main() -> int:
    configure_logging()
    paper = PAPER.read_text(encoding="utf-8")
    defined = _defined(NUMBERS.read_text(encoding="utf-8"))
    body = _body(paper)

    failures = 0
    undefined = sorted(_used(body) - defined)
    if undefined:
        failures += len(undefined)
        _log.error("%d macros used but never defined: %s", len(undefined), undefined)

    elsewhere = set(re.findall(r"\{\{(rs[A-Za-z]+)\}\}", TEMPLATE.read_text(encoding="utf-8")))
    unused = sorted(defined - _used(body) - elsewhere)
    if unused:
        failures += len(unused)
        _log.error("%d macros defined but used nowhere: %s", len(unused), unused)

    bare = _bare_numerals(body)
    if bare:
        failures += len(bare)
        _log.error("%d bare numerals in claim positions:", len(bare))
        for number, token, line in bare:
            _log.error("  line %d: %r in %s", number, token, line[:110])

    if failures:
        _log.error(
            "FAIL: %d problems. Every figure in the paper must resolve to an artifact.", failures
        )
        return 1
    _log.info("PASS: %d macros defined, all used, no bare numeral in any claim position.",
              len(defined))
    return 0


if __name__ == "__main__":
    sys.exit(main())
