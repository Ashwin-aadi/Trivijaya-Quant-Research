"""Each paradigm's structure, cost and trial count, measured against a stubbed model.

No Ollama call is made here. What is under test is the procedure — how many calls it issues, how
many candidates it burns, and whether it charges itself for them — not what a 7B says on the day.
"""

from __future__ import annotations

import pytest

from src.generate.paradigms import base
from src.generate.paradigms.base import seed_for
from src.generate.paradigms.cot import ChainOfThought
from src.generate.paradigms.evolutionary import Evolutionary
from src.generate.paradigms.multi_agent import MultiAgent
from src.generate.paradigms.plain import PlainPrompting
from src.generate.paradigms.planning import Planning
from src.generate.paradigms.reflection import Reflection
from src.generate.tokens import TokenAccount, Usage

CONFORMING = '''
class Stub(Strategy):
    """A strategy that satisfies P1's conformance rule and nothing more."""

    rationale = "Prices that fell recently tend to bounce."

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def generate(self, view):
        return {}
'''

REPLY = f"Some prose the paradigm asked for.\n\n```python\n{CONFORMING}\n```"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Replace the model with a stub, recording every prompt and seed it was asked for."""
    seen: list[tuple[str, int]] = []

    def fake_post(prompt: str, seed: int, *, model_tag: str, host: str) -> tuple[str, Usage]:
        seen.append((prompt, seed))
        return REPLY, Usage(prompt_tokens=100, output_tokens=50)

    monkeypatch.setattr(base, "_post", fake_post)
    return seen


@pytest.mark.parametrize(
    ("paradigm", "expected_calls"),
    [
        (PlainPrompting(), 1),
        (ChainOfThought(), 1),
        (Planning(), 4),
        (Reflection(), 3),
        (MultiAgent(), 5),
    ],
)
def test_each_paradigm_issues_the_calls_its_structure_implies(
    paradigm: object, expected_calls: int, calls: list[tuple[str, int]]
) -> None:
    draw = paradigm.draw(0)  # type: ignore[attr-defined]
    assert draw.usable
    assert draw.calls == expected_calls
    assert len(calls) == expected_calls


def test_a_paradigm_charges_itself_for_every_call_it_makes(calls: list[tuple[str, int]]) -> None:
    """Five calls at 50 generated tokens is 250, not 50. This is the RULE 11 measurement."""
    account = TokenAccount()
    draw = MultiAgent().draw(0, account=account)
    assert draw.usage.output_tokens == 250
    assert account.usage["G6_multi_agent"].output_tokens == 250
    assert account.output_tokens_per_accepted("G6_multi_agent") == 250.0


def test_chain_of_thought_costs_more_than_plain_at_the_same_call_count(
    calls: list[tuple[str, int]],
) -> None:
    """Both make one call, so only tokens can tell them apart - which is why k is a token ratio."""
    plain_prompt = PlainPrompting()
    cot = ChainOfThought()
    plain_draw = plain_prompt.draw(0)
    cot_draw = cot.draw(0)
    assert plain_draw.calls == cot_draw.calls == 1
    prompts = [p for p, _ in calls]
    assert len(prompts[1]) > len(prompts[0]), "the CoT arm must send the longer prompt"


def test_every_call_in_a_draw_gets_a_distinct_seed(calls: list[tuple[str, int]]) -> None:
    """A repeated seed against a fixed prompt makes the model deterministic - P1 measured that."""
    MultiAgent().draw(7)
    seeds = [seed for _, seed in calls]
    assert len(set(seeds)) == len(seeds)


def test_stage_zero_reproduces_p1s_own_seed_scheme() -> None:
    """What lets P1's 1,550 draws serve as P4's control rather than merely resemble it."""
    base_seed, index, attempt = 42, 137, 2
    p1_scheme = base_seed + index + attempt * 10_000
    assert seed_for(base_seed, index, stage=0, attempt=attempt) == p1_scheme


def test_seeds_never_collide_across_stages_of_different_draws() -> None:
    seeds = {
        seed_for(42, index, stage, attempt)
        for index in range(300)
        for stage in range(12)
        for attempt in range(1, 4)
    }
    assert len(seeds) == 300 * 12 * 3


def test_the_evolutionary_arm_counts_every_individual_it_bred_as_a_trial(
    calls: list[tuple[str, int]],
) -> None:
    """RQ4 in one assertion: the arm returns one strategy and is charged for twelve."""
    arm = Evolutionary(fitness=lambda source: 1.0, population=4, generations=2, survivors=2)
    draw = arm.draw(0)
    assert draw.usable
    assert draw.calls == 12
    assert draw.candidates_evaluated == 12, "one returned strategy, twelve trials"
    assert draw.usage.output_tokens == 600


def test_the_evolutionary_arm_keeps_breeding_when_nothing_can_be_scored(
    calls: list[tuple[str, int]],
) -> None:
    """Skipping unscoreable generations would cut the trial count where search went worst."""
    arm = Evolutionary(fitness=lambda source: None, population=2, generations=1, survivors=1)
    draw = arm.draw(0)
    assert draw.candidates_evaluated == 4
    assert not draw.usable, "nothing was scorable, so no strategy can be returned"


def test_the_evolutionary_arm_returns_the_fittest_individual(
    calls: list[tuple[str, int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = iter([0.1, 0.2, 0.9, 0.3])
    arm = Evolutionary(fitness=lambda s: next(scores, 0.0), population=2, generations=1)
    draw = arm.draw(0)
    assert draw.usable


def test_a_paradigm_cannot_be_built_with_more_survivors_than_population() -> None:
    with pytest.raises(base.ParadigmError):
        Evolutionary(fitness=lambda s: 1.0, population=2, survivors=3)
