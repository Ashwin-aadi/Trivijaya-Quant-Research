"""The two arms that replaced multi-agent and evolutionary, and the properties they must have.

No model is called: `_post` is patched, so these test the paradigms' structure and accounting rather
than the model's output. What the model says is the experiment; how the search spends its budget and
counts its trials is the harness, and only the harness can be tested in advance.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.generate.paradigms import graph_of_thoughts as got_module
from src.generate.paradigms import mcts as mcts_module
from src.generate.paradigms.base import ParadigmError
from src.generate.paradigms.graph_of_thoughts import GraphOfThoughts
from src.generate.paradigms.mcts import DECISION_LAYERS, MonteCarloTreeSearch, _floor
from src.generate.paradigms.registry import ARMS, build
from src.generate.tokens import TokenAccount, Usage

#: Identical in shape to `tests/generate/test_paradigms.py`'s fixture: the minimum that satisfies
#: P1's conformance rule, so acceptance here is acceptance there.
CONFORMING = '''
class ProbeStrategy(Strategy):
    """A strategy that satisfies P1's conformance rule and nothing more."""

    rationale = "Prices that fell recently tend to bounce."

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def generate(self, view):
        return {}
'''


class _Recorder:
    """Stands in for the model, remembering every prompt it was asked."""

    def __init__(self, reply: str = "a thought") -> None:
        self.prompts: list[str] = []
        self.seeds: list[int] = []
        self.reply = reply

    def __call__(self, prompt: str, seed: int, **_: Any) -> tuple[str, Usage]:
        self.prompts.append(prompt)
        self.seeds.append(seed)
        return self.reply, Usage(prompt_tokens=10, output_tokens=20)


def _variant_post(replies: Any) -> Any:
    """A model that stamps a serial number into each strategy, so the best one is nameable."""

    def post(prompt: str, seed: int, **_: Any) -> tuple[str, Usage]:
        n = next(replies)
        if "Write the finished strategy" in prompt:
            return f"```python\n# variant {n}\n{CONFORMING}\n```", Usage(10, 20)
        return f"decision {n}", Usage(10, 20)

    return post


@pytest.fixture
def code_recorder() -> _Recorder:
    return _Recorder(reply=f"```python\n{CONFORMING}\n```")


# --- graph of thoughts -----------------------------------------------------------------------


def test_the_three_proposals_are_independent_and_never_see_each_other(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    """The property that distinguishes a graph from the chain it replaced.

    If any of the first three prompts contained an earlier proposal, this arm would be the
    multi-agent chain under a different name and its comparison against it would be meaningless.
    """
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    GraphOfThoughts().draw(0)

    first_three = code_recorder.prompts[:3]
    assert len(set(first_three)) == 1, "the three proposals must ask exactly the same question"
    for prompt in first_three:
        assert "DESIGN 1" not in prompt
        assert code_recorder.reply not in prompt


def test_both_aggregations_see_all_three_proposals(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    """In-degree greater than one is what makes the structure a graph rather than a tree."""
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    GraphOfThoughts().draw(0)

    agreement, strength = code_recorder.prompts[3], code_recorder.prompts[4]
    for prompt in (agreement, strength):
        assert prompt.count("DESIGN") == 3
    assert agreement != strength, "the two aggregations must apply different criteria"


def test_the_graph_arm_makes_seven_calls_and_charges_all_of_them(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    account = TokenAccount()
    draw = GraphOfThoughts().draw(0, account=account)

    assert draw.calls == 7
    assert account.calls["G6_graph_of_thoughts"] == 7
    assert account.usage["G6_graph_of_thoughts"].output_tokens == 7 * 20


def test_no_graph_prompt_names_the_frozen_stack() -> None:
    """The circularity guard, per-module. `scripts/check_paradigm_prompts.py` is the global one."""
    banned = ("audit", "leak", "lookahead", "overfit", "fragility", "capacity", "survivorship")
    constants = [v for k, v in vars(got_module).items() if k.isupper() and isinstance(v, str)]
    assert constants, "the module should expose its prompts as constants"
    for text in constants:
        assert not [w for w in banned if w in text.lower()]


# --- Monte Carlo tree search -----------------------------------------------------------------


def test_every_iteration_completes_a_strategy_and_is_charged_as_a_trial(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    """RQ4's whole mechanism.

    A version of this arm that returned one trial per returned strategy would answer *"does honest
    trial accounting erase the gains of iterating"* dishonestly and by construction. Twelve
    iterations means twelve completed strategies, and the ledger must increment by twelve.
    """
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    draw = MonteCarloTreeSearch(lambda _: 1.0, iterations=12).draw(0)

    assert draw.candidates_evaluated == 12
    # Twelve expansions plus twelve completions, each completion conforming on the first attempt.
    assert draw.calls == 24


def test_an_unscoreable_rollout_is_credited_the_worst_score_seen_not_zero() -> None:
    """Sharpe is routinely negative, so zero is not the bottom of the scale.

    Crediting zero would make a branch that cannot produce runnable code look *better* than one
    that produces losing strategies, and the search would walk straight into it.
    """
    assert _floor([-2.0, -1.0]) == -2.0
    assert _floor([]) == 0.0


def test_an_unscoreable_completion_never_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draw the fitness declined to score must not be returned over one it scored."""
    monkeypatch.setattr("src.generate.paradigms.base._post", _variant_post(iter(range(100))))
    scores: dict[int, float | None] = {1: -3.0, 3: None, 5: -1.0, 7: None}

    def fitness(source: str) -> float | None:
        return scores[int(source.split("# variant ")[1].split("\n")[0])]

    draw = MonteCarloTreeSearch(fitness, iterations=4).draw(0)
    assert draw.source.startswith("# variant 5"), "the best *scored* completion must be returned"


def test_the_search_returns_the_highest_scoring_completion(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.generate.paradigms.base._post", _variant_post(iter(range(100))))
    # Fitness rises with the serial number stamped into the source, so the best is identifiable.
    draw = MonteCarloTreeSearch(
        lambda src: float(src.split("# variant ")[1].split("\n")[0]), iterations=4
    ).draw(0)

    # Completions are the even-indexed calls: 1, 3, 5, 7.
    assert draw.source.startswith("# variant 7")


def test_a_draw_whose_fitness_never_returns_a_number_is_still_judged_by_the_shared_rule(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    """Otherwise this arm alone would be held to the fitness function's stricter standard.

    Yield is *"executes and takes a position"*, measured downstream by the same backtest for every
    arm. If a search that produced perfectly conforming code reported an unusable draw merely
    because its own scorer declined to score it, this arm's yield would not be comparable to any
    other's.
    """
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    draw = MonteCarloTreeSearch(lambda _: None, iterations=3).draw(0)

    assert draw.usable
    assert "ProbeStrategy" in draw.source


def test_the_tree_never_grows_deeper_than_the_decision_layers(
    monkeypatch: pytest.MonkeyPatch, code_recorder: _Recorder
) -> None:
    """A path deeper than the layers would ask the model to decide something undefined."""
    monkeypatch.setattr("src.generate.paradigms.base._post", code_recorder)
    search = MonteCarloTreeSearch(lambda _: 1.0, iterations=20, branching=1)
    search.draw(0)

    decision_prompts = [p for p in code_recorder.prompts if "Decide " in p]
    assert len(decision_prompts) <= 20
    for layer in DECISION_LAYERS:
        assert any(layer in p for p in decision_prompts)


def test_a_zero_iteration_search_is_refused() -> None:
    with pytest.raises(ParadigmError):
        MonteCarloTreeSearch(lambda _: 1.0, iterations=0)


def test_no_mcts_prompt_names_the_frozen_stack() -> None:
    banned = ("audit", "leak", "lookahead", "overfit", "fragility", "capacity", "survivorship")
    constants = [v for k, v in vars(mcts_module).items() if k.isupper() and isinstance(v, str)]
    for text in constants:
        assert not [w for w in banned if w in text.lower()]


# --- the registry ----------------------------------------------------------------------------


def test_the_experiment_is_exactly_six_arms_and_the_retired_two_are_not_among_them() -> None:
    assert list(ARMS) == ["G1", "G2", "G4", "G5", "G6", "G7"]
    assert ARMS["G6"] == "G6_graph_of_thoughts"
    assert ARMS["G7"] == "G7_mcts"
    assert "multi_agent" not in ARMS.values()
    assert "G7_evolutionary" not in ARMS.values()


def test_an_unknown_arm_is_refused_rather_than_silently_skipped() -> None:
    with pytest.raises(ParadigmError):
        build("G3")


@pytest.mark.parametrize("short", ["G1", "G2", "G4", "G5", "G6"])
def test_only_the_search_arm_builds_a_fitness_worker(short: str) -> None:
    """Building any other arm must not start a process holding the price panel."""
    paradigm, fitness = build(short)
    assert fitness is None
    assert paradigm.name == ARMS[short]
