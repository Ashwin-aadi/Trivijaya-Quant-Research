"""Inter-rater agreement for the semantic audit labels: Cohen's kappa and the table behind it.

Single responsibility: given two independent sequences of categorical labels over the same items,
report how far the two raters agree beyond what their individual labelling habits would produce
by chance.

Raw agreement on its own is evidence of very little. If one rater marks 90% of a corpus
``consistent`` and the other marks 92% of it ``consistent``, they will agree on most items while
having demonstrated no shared judgment at all — two raters answering independently at those
frequencies would also agree on most items. Cohen's kappa (Cohen 1960, "A coefficient of agreement
for nominal scales", Educational and Psychological Measurement 20(1), 37-46) subtracts that
expected chance agreement::

    kappa = (p_o - p_e) / (1 - p_e)

``p_o`` is the observed fraction of items the raters labelled identically. ``p_e`` is the fraction
they would be expected to agree on if each assigned labels independently at their own observed
marginal rates.

Landis & Koch (1977), "The measurement of observer agreement for categorical data", Biometrics
33(1), 159-174, give the conventional reading:

===============  ==================
kappa            interpretation
===============  ==================
below 0.00       poor
0.00 to 0.20     slight
0.21 to 0.40     fair
0.41 to 0.60     moderate
0.61 to 0.80     substantial
0.81 to 1.00     almost perfect
===============  ==================

Those bands are a convention, not a test. They carry no p-value and no sample size, so a kappa
from this module is only ever reported next to its ``n``.

**Kappa is undefined when p_e equals 1**, which happens exactly when both raters used one and the
same single category on every item. Observed agreement is then perfect and completely
uninformative, and the arithmetic reduces to ``(1 - 1) / (1 - 1)``. That case returns
:data:`UNDEFINED_KAPPA` rather than dividing by zero, and rather than returning some number chosen
to stand in for a statistic that does not exist. It is a real possibility here and not a corner
case: a local model that answers ``consistent`` to everything, audited against a reviewer who also
found nothing wrong, produces exactly this table.

Implemented from the definition rather than pulled from a library. It is a dozen lines of
arithmetic, and it is the number that decides whether the semantic auditor stays in the paper.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: Returned when kappa does not exist for the data supplied. NaN and not 0.0: zero is a real
#: result meaning "no better than chance", and conflating the two would report an unmeasurable
#: comparison as a measured failure.
UNDEFINED_KAPPA: float = math.nan

#: Chance agreement this close to 1 leaves no denominator to divide by. A tolerance rather than an
#: equality test because ``p_e`` is a sum of products of floats: a table that is mathematically
#: degenerate can land on 0.9999999999999998 instead of exactly 1.0, and dividing by that residue
#: yields a kappa in the region of 1e15 that looks like a number.
_MIN_CHANCE_HEADROOM: float = 1e-12


@dataclass(frozen=True)
class CategoryCounts:
    """How often each rater reached for one category, and how often they did so together.

    ``agreed`` is the diagonal cell. Kept per category because a poor overall kappa usually comes
    from one class, and a starved class (both counts near zero) and a contested class (both counts
    high, ``agreed`` low) are different problems with different remedies.
    """

    category: str
    count_a: int
    count_b: int
    agreed: int


@dataclass(frozen=True)
class AgreementSummary:
    """Kappa with the context needed to read it: raw agreement, the per-class counts, and ``n``."""

    kappa: float
    raw_agreement: float
    per_category: tuple[CategoryCounts, ...]
    n: int


def _validate(
    labels_a: Sequence[str], labels_b: Sequence[str], categories: Sequence[str]
) -> None:
    """Reject any input for which an agreement statistic would be meaningless.

    Everything here raises rather than coercing. A label outside the taxonomy is a bug in whatever
    produced it — a truncated model reply, a typo in a hand-filled sheet — and quietly folding it
    into some nearest category would move the very number this module exists to measure.
    """
    if not categories:
        raise ValueError("categories must not be empty")
    if len(set(categories)) != len(categories):
        raise ValueError(f"categories contains duplicates: {list(categories)}")
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"the two raters labelled different numbers of items: "
            f"{len(labels_a)} vs {len(labels_b)}; agreement is only defined pairwise"
        )
    if not labels_a:
        raise ValueError("cannot measure agreement over zero items")
    known = set(categories)
    for rater, labels in (("a", labels_a), ("b", labels_b)):
        unknown = sorted(set(labels) - known)
        if unknown:
            raise ValueError(
                f"rater {rater} used labels outside the agreed categories: {unknown}; "
                f"allowed categories are {list(categories)}"
            )


def confusion_matrix(
    labels_a: Sequence[str], labels_b: Sequence[str], categories: Sequence[str]
) -> list[list[int]]:
    """Counts of every (rater A label, rater B label) pair, indexed in ``categories`` order.

    Row ``i`` is what rater A called ``categories[i]``; column ``j`` is what rater B called it. The
    diagonal is agreement, everything off it is a disagreement, and the asymmetry between cell
    ``(i, j)`` and cell ``(j, i)`` shows which rater is the more suspicious of the two. Reported
    alongside kappa and not in place of it, because one number cannot say *where* the raters part
    company, and that is usually the more useful finding.
    """
    _validate(labels_a, labels_b, categories)
    position = {category: index for index, category in enumerate(categories)}
    table = [[0] * len(categories) for _ in categories]
    for label_a, label_b in zip(labels_a, labels_b, strict=True):
        table[position[label_a]][position[label_b]] += 1
    return table


def _kappa_from_table(table: list[list[int]], n: int) -> float:
    """Cohen's kappa computed from an already-built confusion matrix.

    Split out so :func:`cohens_kappa` and :func:`agreement_summary` share one implementation and
    cannot drift into reporting two different numbers for the same data.
    """
    size = len(table)
    observed = sum(table[i][i] for i in range(size)) / n
    # Expected agreement under independence: for each category, the chance rater A picks it times
    # the chance rater B picks it, using each rater's own observed rate.
    expected = 0.0
    for i in range(size):
        used_by_a = sum(table[i])
        used_by_b = sum(table[row][i] for row in range(size))
        expected += (used_by_a / n) * (used_by_b / n)
    if 1.0 - expected <= _MIN_CHANCE_HEADROOM:
        return UNDEFINED_KAPPA
    return (observed - expected) / (1.0 - expected)


def cohens_kappa(
    labels_a: Sequence[str], labels_b: Sequence[str], categories: Sequence[str]
) -> float:
    """Chance-corrected agreement between two raters over the same items.

    Ranges from 1.0 (identical labelling) down through 0.0 (exactly what independent raters with
    these habits would manage) to negative values (the raters disagree more than chance predicts,
    which means systematic opposition rather than noise). See the module docstring for the Landis
    & Koch bands and for the case in which the statistic does not exist.
    """
    table = confusion_matrix(labels_a, labels_b, categories)
    return _kappa_from_table(table, len(labels_a))


def agreement_summary(
    labels_a: Sequence[str], labels_b: Sequence[str], categories: Sequence[str]
) -> AgreementSummary:
    """Kappa, raw agreement, per-category counts, and the sample size all together.

    The sample size is carried in the return value rather than left to the caller because the
    charter forbids reporting a figure without one, and a kappa over 50 hand-labelled items is a
    materially weaker claim than the same kappa over a thousand.
    """
    table = confusion_matrix(labels_a, labels_b, categories)
    n = len(labels_a)
    size = len(categories)
    counts = tuple(
        CategoryCounts(
            category=category,
            count_a=sum(table[index]),
            count_b=sum(table[row][index] for row in range(size)),
            agreed=table[index][index],
        )
        for index, category in enumerate(categories)
    )
    return AgreementSummary(
        kappa=_kappa_from_table(table, n),
        raw_agreement=sum(table[i][i] for i in range(size)) / n,
        per_category=counts,
        n=n,
    )
