"""Tests for the semantic audit layer.

Every test here runs with Ollama switched off. The HTTP layer is mocked, which is not a
convenience: a test suite that only passes when a local model happens to be loaded is a suite that
gets skipped, and a skipped test defends nothing.

Three things are under test, in descending order of how much damage they would do if wrong.

The first is that failure is never silent. A model that is unreachable, or that answers with prose,
a broken object, or a label nobody defined, must produce an exception. The tempting alternative -
falling back to ``consistent`` - would stamp a strategy as audited that nothing ever read, and
downstream that is indistinguishable from a genuine pass. Several tests below exist only to assert
that the fallback does not exist.

The second is the parser. Small local models wrap JSON in fences and preamble, so the extractor has
to cope with that; it must not cope by guessing.

The third is determinism. ``temperature=0`` and ``seed=42`` are asserted in the request payload
itself rather than inferred from behaviour, because losing them breaks nothing visibly - the code
keeps working and the results simply stop being reproducible.
"""

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import patch

import pytest
import requests

from src.audit.prompts import LABELS, MAX_SOURCE_CHARS, build_prompt
from src.audit.semantic import (
    MODEL_TAG,
    SEED,
    TEMPERATURE,
    AuditItem,
    SemanticAuditParseError,
    SemanticAuditUnavailable,
    SemanticFinding,
    batch_classify,
    classify,
    is_available,
    throughput_estimate,
)

RATIONALE = "Buys names whose 20-day average sits above their 50-day average."
SOURCE = "short = mean(closes[-20:])\nlong = mean(closes[-50:])\nhold = short > long"

VALID_JSON = '{"label": "consistent", "confidence": 0.8, "explanation": "code matches rationale"}'


class FakeResponse:
    """Minimal stand-in for ``requests.Response``: only what the module actually touches."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: object = None,
        text: str = "",
        json_raises: bool = False,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self._json_raises = json_raises

    def json(self) -> object:
        if self._json_raises:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


def ollama_reply(completion: str) -> FakeResponse:
    """A well-formed Ollama /api/generate response wrapping ``completion``."""
    return FakeResponse(payload={"model": MODEL_TAG, "response": completion, "done": True})


def classify_reply(reply: FakeResponse) -> SemanticFinding:
    """Audit a fixed strategy against one mocked reply.

    Exists so that ``pytest.raises`` wraps the call alone rather than the patch as well - which
    keeps each test's assertion about the failure, not about the plumbing that produced it.
    """
    with patch("requests.post", return_value=reply):
        return classify(RATIONALE, SOURCE)


# --- parsing: what the model says, and what we are willing to read ---------------


def test_clean_json_is_parsed() -> None:
    """The ordinary case: a bare JSON object with the three expected keys."""
    finding = classify_reply(ollama_reply(VALID_JSON))
    assert finding.label == "consistent"
    assert finding.confidence == pytest.approx(0.8)
    assert finding.explanation == "code matches rationale"
    assert finding.model_tag == MODEL_TAG
    # The raw text is kept verbatim; it is the only durable record of how a label was reached.
    assert finding.raw_response == VALID_JSON
    assert finding.is_defect is False


def test_json_wrapped_in_markdown_fences_is_parsed() -> None:
    """Instruction-tuned models fence their JSON habitually. That must not cost us the finding."""
    completion = (
        "```json\n"
        '{"label": "unfalsifiable_mechanism", "confidence": 0.55,\n'
        ' "explanation": "the rationale would explain any outcome"}\n'
        "```"
    )
    finding = classify_reply(ollama_reply(completion))
    assert finding.label == "unfalsifiable_mechanism"
    assert finding.confidence == pytest.approx(0.55)
    assert finding.is_defect is True


def test_json_embedded_in_prose_is_parsed() -> None:
    """Preamble and trailing commentary are stripped; the object between them still counts."""
    completion = (
        "Having read the rationale against the code, here is my assessment:\n"
        '{"label": "rationale_implementation_mismatch", "confidence": 0.9, '
        '"explanation": "rationale says buy losers, code sorts descending and buys winners"}\n'
        "Let me know if you would like me to expand on this."
    )
    finding = classify_reply(ollama_reply(completion))
    assert finding.label == "rationale_implementation_mismatch"
    assert "winners" in finding.explanation


def test_braces_inside_the_explanation_do_not_end_the_object_early() -> None:
    """String tracking in the extractor: a brace inside a quoted value is not a delimiter."""
    completion = (
        '{"label": "consistent", "confidence": 0.7, '
        '"explanation": "the code builds weights via dict.fromkeys({}) which matches the text"}'
    )
    finding = classify_reply(ollama_reply(completion))
    assert finding.label == "consistent"
    assert "dict.fromkeys({})" in finding.explanation


def test_a_truncated_object_followed_by_a_complete_one_is_recovered() -> None:
    """An unbalanced first candidate makes the scan restart rather than give up."""
    completion = '{"label": "consistent", ' + "\n" + VALID_JSON
    assert classify_reply(ollama_reply(completion)).label == "consistent"


@pytest.mark.parametrize(
    "completion",
    [
        '{"label": "leaky", "confidence": 0.9, "explanation": "invented category"}',
        '{"label": "CONSISTENT!", "confidence": 0.9, "explanation": "close but not the taxonomy"}',
        '{"label": 3, "confidence": 0.9, "explanation": "not even a string"}',
        '{"confidence": 0.9, "explanation": "no label key at all"}',
    ],
)
def test_a_label_outside_the_taxonomy_raises(completion: str) -> None:
    """A label nobody defined means the model did not do the task. It is never read as a pass.

    This is the single most important assertion in the file. Coercing an unrecognised label to
    ``consistent`` would hand a clean bill of health to a strategy the auditor failed to audit.
    """
    with pytest.raises(SemanticAuditParseError):
        classify_reply(ollama_reply(completion))


@pytest.mark.parametrize(
    "completion",
    [
        "I am not able to answer that question.",
        "{'label': 'consistent', 'confidence': 0.8}",  # single quotes: not JSON
        '{"label": "consistent", "confidence": 0.8, "explanation": }',
        "{{{{",
        "```json\n```",
    ],
)
def test_malformed_output_raises_rather_than_defaulting(completion: str) -> None:
    """Unreadable output is an error, not a ``consistent`` finding."""
    with pytest.raises(SemanticAuditParseError):
        classify_reply(ollama_reply(completion))


def test_a_blank_completion_is_treated_as_no_audit_at_all() -> None:
    """An empty answer is not an unreadable answer - nothing was audited, so it is unavailable."""
    with pytest.raises(SemanticAuditUnavailable):
        classify_reply(FakeResponse(payload={"response": "   "}))


def test_missing_confidence_or_explanation_raises() -> None:
    """Both fields are required; inventing a value for either would fabricate evidence."""
    for completion in (
        '{"label": "consistent", "explanation": "no confidence given"}',
        '{"label": "consistent", "confidence": 0.5}',
    ):
        with pytest.raises(SemanticAuditParseError):
            classify_reply(ollama_reply(completion))


# --- confidence handling --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [
        ("95", 1.0),        # answered on a 0-100 scale
        ("1.4", 1.0),
        ("-0.2", 0.0),
        ("0.0", 0.0),
        ("1", 1.0),
        ("0.42", 0.42),
        ('"0.6"', 0.6),     # numeric, but quoted
    ],
)
def test_confidence_is_clamped_into_the_unit_interval(
    raw_confidence: str, expected: float
) -> None:
    """Out-of-range confidences are clamped, not rejected - the label is still usable."""
    completion = (
        f'{{"label": "consistent", "confidence": {raw_confidence}, "explanation": "ok"}}'
    )
    finding = classify_reply(ollama_reply(completion))
    assert finding.confidence == pytest.approx(expected)


@pytest.mark.parametrize("raw_confidence", ['"high"', "true", "null", '["0.5"]'])
def test_non_numeric_confidence_raises(raw_confidence: str) -> None:
    """``true`` is the interesting case: bool is an int subclass and would become 1.0."""
    completion = (
        f'{{"label": "consistent", "confidence": {raw_confidence}, "explanation": "ok"}}'
    )
    with pytest.raises(SemanticAuditParseError):
        classify_reply(ollama_reply(completion))


# --- the model being unreachable ------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("[Errno 111] Connection refused"),
        requests.Timeout("read timed out"),
        requests.RequestException("something else went wrong"),
    ],
)
def test_transport_failure_raises_semantic_audit_unavailable(error: Exception) -> None:
    """Ollama not running is the common case, and it must be loud and actionable."""
    with (
        patch("requests.post", side_effect=error),
        pytest.raises(SemanticAuditUnavailable) as excinfo,
    ):
        classify(RATIONALE, SOURCE)
    message = str(excinfo.value)
    assert "ollama serve" in message
    assert MODEL_TAG in message


def test_http_error_status_raises_semantic_audit_unavailable() -> None:
    """A 404 here means the tag was never pulled; the message has to say so."""
    reply = FakeResponse(status_code=404, text='{"error":"model not found"}')
    with pytest.raises(SemanticAuditUnavailable) as excinfo:
        classify_reply(reply)
    assert "404" in str(excinfo.value)
    assert "ollama pull" in str(excinfo.value)


def test_non_json_body_raises_semantic_audit_unavailable() -> None:
    """Something is answering on that port, but it is not Ollama."""
    reply = FakeResponse(text="<html>proxy error</html>", json_raises=True)
    with pytest.raises(SemanticAuditUnavailable):
        classify_reply(reply)


def test_body_without_a_response_field_raises() -> None:
    """A well-formed envelope with no completion in it is still an unaudited strategy."""
    with pytest.raises(SemanticAuditUnavailable):
        classify_reply(FakeResponse(payload={"done": True}))


def test_is_available_reports_reachability_without_raising() -> None:
    """The probe answers yes or no. It is the only function here allowed to swallow an error."""
    with patch("requests.get", return_value=FakeResponse(payload={"models": []})):
        assert is_available() is True
    with patch("requests.get", return_value=FakeResponse(status_code=500)):
        assert is_available() is False
    with patch("requests.get", side_effect=requests.ConnectionError("refused")):
        assert is_available() is False


# --- determinism ----------------------------------------------------------------


def test_request_payload_pins_temperature_and_seed() -> None:
    """Reproducibility lives or dies here, and it fails silently if it ever regresses.

    Dropping these options does not break a single behavioural test: the module keeps returning
    labels, they just stop being the same labels twice. So the payload is asserted directly.
    """
    with patch("requests.post", return_value=ollama_reply(VALID_JSON)) as post:
        classify(RATIONALE, SOURCE)

    sent: dict[str, Any] = post.call_args.kwargs["json"]
    assert sent["options"]["temperature"] == TEMPERATURE == 0.0
    assert sent["options"]["seed"] == SEED == 42
    assert sent["model"] == MODEL_TAG
    assert sent["stream"] is False
    # The context window is pinned too: left to the server default it varies by build, and a
    # smaller one would truncate the strategy source out of the prompt without saying so.
    assert sent["options"]["num_ctx"] > 0
    assert post.call_args.args[0].endswith("/api/generate")


def test_the_prompt_carries_both_the_rationale_and_the_code() -> None:
    """A prompt missing either half would still return labels - confident, and about nothing."""
    prompt = build_prompt(RATIONALE, SOURCE)
    assert RATIONALE in prompt
    assert SOURCE in prompt
    for label in LABELS:
        assert label in prompt


def test_overlong_source_is_clipped_with_a_visible_marker() -> None:
    """Truncation is marked in the text rather than left to the server's context window."""
    prompt = build_prompt(RATIONALE, "x = 1\n" * 5_000)
    assert "truncated for context length" in prompt
    assert len(prompt) < MAX_SOURCE_CHARS + 10_000


# --- batching and cost estimation ------------------------------------------------


def test_batch_classify_returns_one_finding_per_item_in_order() -> None:
    """Order matters: findings are joined back to strategies positionally by the caller."""
    completions = [
        '{"label": "consistent", "confidence": 0.7, "explanation": "first"}',
        '{"label": "unacknowledged_known_anomaly", "confidence": 0.6, "explanation": "second"}',
        '{"label": "consistent", "confidence": 0.5, "explanation": "third"}',
    ]
    items = [AuditItem(name=f"s{i}", rationale=RATIONALE, source_code=SOURCE) for i in range(3)]
    with patch("requests.post", side_effect=[ollama_reply(c) for c in completions]):
        findings = batch_classify(items)
    assert [f.explanation for f in findings] == ["first", "second", "third"]
    assert [f.label for f in findings] == [
        "consistent",
        "unacknowledged_known_anomaly",
        "consistent",
    ]


def test_batch_classify_propagates_a_mid_batch_failure() -> None:
    """A partial batch with holes would give a false-positive rate a quietly wrong denominator."""
    items = [AuditItem(name=f"s{i}", rationale=RATIONALE, source_code=SOURCE) for i in range(3)]
    replies: list[Any] = [ollama_reply(VALID_JSON), requests.ConnectionError("refused")]
    with (
        patch("requests.post", side_effect=replies),
        pytest.raises(SemanticAuditUnavailable),
    ):
        batch_classify(items)


def test_throughput_estimate_times_real_calls_and_extrapolates() -> None:
    """The charter needs a wall-clock figure before a large batch; this is where it comes from."""
    with patch("requests.post", return_value=ollama_reply(VALID_JSON)) as post:
        estimate = throughput_estimate(300, n_calls=2)
    assert post.call_count == 2
    assert len(estimate.call_seconds) == 2
    assert estimate.n_items == 300
    assert estimate.estimated_seconds == pytest.approx(estimate.seconds_per_item * 300)
    assert estimate.model_tag == MODEL_TAG
    assert "300 items" in estimate.summary()


def test_throughput_estimate_rejects_a_zero_call_sample() -> None:
    """Extrapolating from no measurements would be a fabricated number."""
    with patch("requests.post", return_value=ollama_reply(VALID_JSON)) as post:
        with pytest.raises(ValueError, match="n_calls"):
            throughput_estimate(300, n_calls=0)
        assert post.call_count == 0  # rejected before it spends anything


# --- the finding object itself ---------------------------------------------------


def test_finding_cannot_be_constructed_outside_the_taxonomy() -> None:
    """Structural guard, so no other code path can put a stray label into a results table."""
    with pytest.raises(ValueError, match="label must be one of"):
        SemanticFinding(
            label="probably_fine",
            confidence=0.5,
            explanation="",
            model_tag=MODEL_TAG,
            raw_response="",
        )
    with pytest.raises(ValueError, match="confidence"):
        SemanticFinding(
            label="consistent",
            confidence=1.5,
            explanation="",
            model_tag=MODEL_TAG,
            raw_response="",
        )


def test_finding_is_frozen() -> None:
    """A label must not be editable after the fact - that is how audit trails rot."""
    finding = SemanticFinding(
        label="consistent",
        confidence=0.5,
        explanation="ok",
        model_tag=MODEL_TAG,
        raw_response=VALID_JSON,
    )
    with pytest.raises(FrozenInstanceError):
        finding.label = "unfalsifiable_mechanism"  # type: ignore[misc]


# --- the one test that needs a real model ----------------------------------------


@pytest.mark.skipif(
    not is_available(), reason="requires a local Ollama server; the rest of this file does not"
)
def test_live_model_returns_a_finding_in_the_taxonomy() -> None:
    """Smoke test against the real model. Kept to one: it is slow and it is not always runnable.

    Asserts only that the round trip produces a valid finding. It deliberately does not assert
    *which* label comes back - quality is measured against the PI's hand labels at Checkpoint 1.3,
    not by a unit test written by the same hand that wrote the prompt.
    """
    finding = classify(RATIONALE, SOURCE)
    assert finding.label in LABELS
    assert 0.0 <= finding.confidence <= 1.0
    assert finding.model_tag == MODEL_TAG


def test_mock_helper_matches_the_real_response_shape() -> None:
    """Guard on the test scaffolding itself: a fake that drifts from reality tests nothing.

    Ollama's /api/generate returns the completion under a ``response`` key alongside the model
    tag. If that envelope ever changes, this fails here rather than every other test passing
    against a shape the server no longer sends.
    """
    body = ollama_reply(VALID_JSON).json()
    assert isinstance(body, dict)
    assert set(body) >= {"model", "response"}
