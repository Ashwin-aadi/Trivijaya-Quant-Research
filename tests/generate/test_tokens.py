"""The token accountant is the only thing standing between P4 and an invalid RULE 11 comparison."""

from __future__ import annotations

from src.generate.tokens import TokenAccount, Usage


def test_usage_adds_componentwise() -> None:
    assert Usage(3, 5) + Usage(7, 11) == Usage(10, 16)


def test_usage_total_is_read_plus_generated() -> None:
    assert Usage(prompt_tokens=40, output_tokens=2).total_tokens == 42


def test_from_ollama_reads_both_counters() -> None:
    body = {"response": "...", "prompt_eval_count": 1200, "eval_count": 340}
    assert Usage.from_ollama(body) == Usage(prompt_tokens=1200, output_tokens=340)


def test_from_ollama_treats_missing_counters_as_zero_rather_than_raising() -> None:
    """Ollama omits them on some error paths. A zero is visibly wrong; a crash loses the draw."""
    assert Usage.from_ollama({"response": ""}) == Usage(0, 0)
    assert Usage.from_ollama({"eval_count": None}) == Usage(0, 0)


def test_failed_draws_are_charged_for_the_tokens_they_burned() -> None:
    """Charging only successes would make the noisiest paradigm look like the cheapest."""
    account = TokenAccount()
    account.record_call("G5", Usage(100, 200))
    account.record_draw("G5", accepted=False)
    account.record_call("G5", Usage(100, 200))
    account.record_draw("G5", accepted=True)

    assert account.usage["G5"].output_tokens == 400
    assert account.draws["G5"] == 2
    assert account.accepted["G5"] == 1
    # 400 generated tokens bought exactly one usable strategy, not two.
    assert account.output_tokens_per_accepted("G5") == 400.0


def test_cost_per_accepted_is_none_when_nothing_was_accepted() -> None:
    """None rather than infinity, so a barren arm cannot be silently averaged into a comparison."""
    account = TokenAccount()
    account.record_call("G7", Usage(10, 20))
    account.record_draw("G7", accepted=False)
    assert account.output_tokens_per_accepted("G7") is None


def test_summary_is_json_serialisable_and_covers_every_named_paradigm() -> None:
    account = TokenAccount()
    account.record_call("G1", Usage(1, 2))
    account.record_draw("G1", accepted=True)
    summary = account.to_dict()
    assert set(summary) == {"G1"}
    assert summary["G1"]["output_tokens"] == 2
    assert summary["G1"]["output_tokens_per_accepted"] == 2.0
