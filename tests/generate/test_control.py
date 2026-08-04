"""The compute-matched control, and the theme-stratification trap it exists to avoid."""

from __future__ import annotations

import pytest

from src.generate.paradigms.base import ParadigmError
from src.generate.paradigms.control import (
    best_of_block,
    matched_k,
    resample_blocks,
    run_control,
)
from src.generate.prompts import THEMES, theme_for


def test_matched_k_rounds_up_so_the_surplus_goes_to_the_control() -> None:
    """Rounding down would hand the treatment a budget the control never received."""
    assert matched_k(2500.0, 1000.0) == 3
    assert matched_k(2000.0, 1000.0) == 2
    assert matched_k(500.0, 1000.0) == 1  # never below one draw


def test_matched_k_is_a_token_ratio_not_a_call_ratio() -> None:
    """Chain of thought makes one call and costs more than one. Calls would miss that entirely."""
    # One call each, but 3.2x the generated tokens: k must be 4, not 1.
    assert matched_k(3200.0, 1000.0) == 4


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_matched_k_refuses_a_nonpositive_budget(bad: float) -> None:
    with pytest.raises(ParadigmError):
        matched_k(1000.0, bad)


def test_blocks_are_contiguous_and_inside_the_corpus() -> None:
    blocks = resample_blocks(corpus_size=1550, k=5, draws=200, seed=42)
    assert len(blocks) == 200
    for block in blocks:
        assert list(block) == list(range(block[0], block[0] + 5))
        assert block[0] >= 0 and block[-1] < 1550


def test_blocks_are_reproducible_under_the_seed() -> None:
    assert resample_blocks(1550, 5, 50, seed=42) == resample_blocks(1550, 5, 50, seed=42)
    assert resample_blocks(1550, 5, 50, seed=42) != resample_blocks(1550, 5, 50, seed=43)


def test_a_contiguous_block_reproduces_the_theme_cycle_of_fresh_draws() -> None:
    """The reason blocks are contiguous rather than uniform.

    P1's corpus is stratified: theme_for(index) is THEMES[index % 12]. k fresh draws would cover k
    consecutive themes, so k resampled draws must too. A uniform sample would give a random mixture
    with repeats - a different sampling scheme, and an invisible confound.
    """
    k = len(THEMES)
    for block in resample_blocks(corpus_size=1550, k=k, draws=20, seed=42):
        themes = [theme_for(i) for i in block]
        assert set(themes) == set(THEMES), "a full-cycle block must cover every theme exactly once"
        assert len(set(themes)) == k


def test_a_block_with_nothing_rankable_is_a_failed_draw_not_a_discarded_one() -> None:
    """Dropping empty blocks would compare the control's best against all of the treatment's."""
    scores: dict[int, float | None] = {0: None, 1: None, 2: None}
    draw = best_of_block((0, 1, 2), scores)
    assert draw.selected is None
    assert draw.score is None
    assert not draw.usable


def test_best_of_block_picks_the_highest_score() -> None:
    draw = best_of_block((0, 1, 2), {0: 0.1, 1: None, 2: 0.9})
    assert draw.selected == 2
    assert draw.score == 0.9


def test_ties_break_to_the_lowest_index_not_to_dictionary_order() -> None:
    """P1's corpus holds 11 clusters of behaviourally identical strategies, so ties are common."""
    assert best_of_block((7, 3, 5), {7: 0.5, 3: 0.5, 5: 0.5}).selected == 3


def test_run_control_yields_one_draw_per_block_including_the_barren_ones() -> None:
    scores: dict[int, float | None] = {i: (0.5 if i % 7 == 0 else None) for i in range(100)}
    draws = run_control(100, scores, k=3, draws=40, seed=42)
    assert len(draws) == 40
    assert any(not d.usable for d in draws), "sparse scores must produce some failed control draws"
    assert any(d.usable for d in draws)


def test_a_block_longer_than_the_corpus_is_refused() -> None:
    with pytest.raises(ParadigmError):
        resample_blocks(corpus_size=4, k=5, draws=1)
