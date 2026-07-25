"""Semantic audit layer: read a strategy's stated rationale against the code that implements it.

The third and least mechanical of the three auditor layers. The static layer reads code and the
statistical layer reads returns; neither can tell that a strategy whose rationale describes buying
oversold names in fact buys the winners, because both halves are individually well-formed. That
mismatch is only visible to something that reads prose and code together, which is what this
module asks a local language model to do.

Three defects are in scope - rationale/implementation mismatch, unfalsifiable mechanism, and a
known anomaly presented as a discovery - plus ``consistent`` for everything else. The taxonomy and
the prompt that teaches it live in :mod:`src.audit.prompts`.

Two properties this module is built around:

*Never guess.* Every failure raises. If the model is unreachable, or returns something that cannot
be parsed, the caller gets an exception, not a ``consistent`` label. A silent default here would
issue a clean bill of health to a strategy nothing ever looked at, and that failure is invisible
downstream: it looks exactly like a strategy that passed.

*Determinism.* Every request pins ``temperature=0`` and ``seed=42``. That makes a rerun reproduce
its labels only as far as the model itself is held fixed - the same model tag, the same
quantization, the same Ollama build. A re-pull of a moving tag can change every label in the
corpus, so the tag used is recorded on each finding and belongs in the run manifest.

Inference is local, through Ollama's HTTP API. No paid service is contacted from this repository.
The model tag, host, context window and timeouts are read from the ``audit:`` section of
config/config.yaml rather than fixed here, so a run manifest's config hash pins them.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import requests

from src.audit.prompts import LABELS, PROBE_RATIONALE, PROBE_SOURCE, build_prompt
from src.common.config import DEFAULT_CONFIG_PATH, AuditConfig, load_config
from src.common.exceptions import LabError
from src.common.log import get_logger

_log = get_logger(__name__)

# Endpoint paths are Ollama's HTTP API, not parameters of this lab, so they stay here rather than
# in config.yaml - changing them would mean talking to a different server, not tuning this one.
GENERATE_PATH: Final[str] = "/api/generate"
TAGS_PATH: Final[str] = "/api/tags"

#: Mirrors the ``audit:`` section of config/config.yaml, which is the source of truth; these values
#: are used only when no config file can be found, so that importing this module - which strategy
#: fixtures and tooling do from arbitrary working directories - never depends on one. The reasoning
#: behind each value lives in config.yaml and is not repeated here. Keeping the two in step is not
#: left to good intentions: tests/common/test_config.py asserts they are identical, so a change to
#: one that is not made to the other fails the suite rather than silently changing which model
#: labelled a corpus.
_FALLBACK_AUDIT: Final[AuditConfig] = AuditConfig(
    model_tag="qwen2.5:7b-instruct-q4_K_M",
    ollama_host="http://localhost:11434",
    num_ctx=4_096,
    request_timeout_seconds=180.0,
    probe_timeout_seconds=2.0,
)


def _resolve_audit_config() -> AuditConfig:
    """Read the audit section from the default config path, falling back only if it is absent.

    The fallback is deliberately narrow. A missing file means this module was imported from
    somewhere without a checkout underfoot, which is not an error. A file that exists but whose
    ``audit:`` section is missing or malformed raises out of :func:`load_config` instead, because
    running a corpus against quietly different parameters than the config on disk describes is
    exactly the kind of unreproducibility this repository exists to detect.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        return _FALLBACK_AUDIT
    return load_config().audit


_AUDIT: Final[AuditConfig] = _resolve_audit_config()

OLLAMA_HOST: Final[str] = _AUDIT.ollama_host

#: Pinned per the charter's model table. The 7B q4_K_M quantization is the largest that leaves
#: headroom on an 8 GB card; nothing bigger may be substituted without PI approval.
MODEL_TAG: Final[str] = _AUDIT.model_tag

#: Sampling is switched off entirely. The seed is redundant at temperature 0 for most backends but
#: is sent anyway: it costs nothing and it removes the question of whether any residual sampling
#: (tie-breaking, backend-specific kernels) is seeded. Neither is configurable: a run of this
#: auditor with sampling on is not a run of this auditor.
TEMPERATURE: Final[float] = 0.0
SEED: Final[int] = 42

#: Context window, pinned. Left to the server default it varies with the Ollama build, and a
#: silently smaller window truncates the strategy source out of the prompt - the model would then
#: label an item it was never shown. Sized to hold the prompt clip limits in src.audit.prompts.
NUM_CTX: Final[int] = _AUDIT.num_ctx

#: A 7B model on CPU can take well over a minute per item. Generous on purpose: a timeout here
#: raises and loses the item, which is more expensive than waiting.
REQUEST_TIMEOUT_SECONDS: Final[float] = _AUDIT.request_timeout_seconds
AVAILABILITY_TIMEOUT_SECONDS: Final[float] = _AUDIT.probe_timeout_seconds


class SemanticAuditUnavailable(LabError):
    """The local model could not be reached, or returned no usable payload.

    Distinct from a parse failure: this means no audit happened at all. Raised rather than
    returning a neutral label, because an unaudited strategy must never be recorded as audited.
    """


class SemanticAuditParseError(LabError):
    """The model replied, but the reply could not be read as a valid finding.

    Raised after the defensive extraction in :func:`_extract_json_object` has already tried. The
    alternative - falling back to ``consistent`` - would silently convert a broken response into a
    pass, so an unreadable answer is an error the caller has to handle.
    """


@dataclass(frozen=True)
class SemanticFinding:
    """One audited strategy: the label, how sure the model was, and the evidence trail.

    ``raw_response`` is retained deliberately. The label is a judgment produced by a model that
    cannot be re-run to bit-identical output across versions, so the text it actually returned is
    the only durable record of how a label was arrived at.
    """

    label: str
    confidence: float
    explanation: str
    model_tag: str
    raw_response: str

    def __post_init__(self) -> None:
        # Structural: a finding carrying a label outside the taxonomy cannot be constructed at
        # all, so no such value can reach a results table by way of some other code path.
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence!r}")

    @property
    def is_defect(self) -> bool:
        """True for any label other than ``consistent``."""
        return self.label != "consistent"


@dataclass(frozen=True)
class AuditItem:
    """A strategy queued for audit. ``name`` exists only so progress logs identify the item."""

    name: str
    rationale: str
    source_code: str


@dataclass(frozen=True)
class ThroughputEstimate:
    """Measured cost of a few real calls, extrapolated to a full batch."""

    model_tag: str
    n_items: int
    call_seconds: tuple[float, ...]
    seconds_per_item: float
    estimated_seconds: float

    def summary(self) -> str:
        """One line suitable for pasting into a checkpoint report."""
        calls = ", ".join(f"{s:.1f}s" for s in self.call_seconds)
        return (
            f"{self.model_tag}: {len(self.call_seconds)} timed calls ({calls}) "
            f"-> {self.seconds_per_item:.1f}s per item, "
            f"{self.estimated_seconds / 60.0:.1f} min for {self.n_items} items"
        )


def is_available(host: str = OLLAMA_HOST, timeout: float = AVAILABILITY_TIMEOUT_SECONDS) -> bool:
    """Report whether an Ollama server answers at ``host``. Never raises.

    The one place in this module that swallows an exception, and it does so because returning
    "no" is precisely this function's job - it is a probe, not an audit, and it produces no label.
    Callers that need an audit call :func:`classify`, which raises instead.
    """
    try:
        response = requests.get(f"{host}{TAGS_PATH}", timeout=timeout)
    except requests.RequestException:
        return False
    return bool(response.status_code == 200)


def classify(
    rationale: str,
    source_code: str,
    *,
    model_tag: str = MODEL_TAG,
    host: str = OLLAMA_HOST,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> SemanticFinding:
    """Audit one strategy's rationale against its source, returning a single labelled finding.

    Raises :class:`SemanticAuditUnavailable` if the model cannot be reached and
    :class:`SemanticAuditParseError` if it answers with something unreadable. It never returns a
    default.
    """
    prompt = build_prompt(rationale, source_code)
    raw = _post_generate(prompt, model_tag=model_tag, host=host, timeout=timeout)
    return _parse_response(raw, model_tag=model_tag)


def batch_classify(
    items: Sequence[AuditItem],
    *,
    model_tag: str = MODEL_TAG,
    host: str = OLLAMA_HOST,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> list[SemanticFinding]:
    """Audit a list of strategies in order, logging progress as it goes.

    A failure on any item propagates rather than being collected. Returning a partial list with
    holes in it invites a caller to compute a false-positive rate over a denominator that quietly
    lost items; the batch is cheap enough to rerun, and a wrong rate is not cheap at all.
    """
    findings: list[SemanticFinding] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        finding = classify(
            item.rationale, item.source_code, model_tag=model_tag, host=host, timeout=timeout
        )
        findings.append(finding)
        _log.info(
            "semantic audit %d/%d: %s -> %s (confidence %.2f)",
            index, len(items), item.name, finding.label, finding.confidence,
        )
    elapsed = time.perf_counter() - started
    _log.info("semantic audit finished: %d items in %.1f s", len(findings), elapsed)
    return findings


def throughput_estimate(
    n_items: int,
    *,
    n_calls: int = 3,
    model_tag: str = MODEL_TAG,
    host: str = OLLAMA_HOST,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> ThroughputEstimate:
    """Time ``n_calls`` real calls on a representative item and extrapolate to ``n_items``.

    Required before any large batch: the charter caps an unreported step at 30 minutes of
    wall-clock, and that cannot be respected without measuring first. The first call includes
    loading the model into VRAM and is kept in the average rather than discarded - that biases the
    estimate upward, which is the correct direction for a figure used to decide whether to halt.
    """
    if n_calls < 1:
        raise ValueError(f"n_calls must be at least 1, got {n_calls}")
    prompt = build_prompt(PROBE_RATIONALE, PROBE_SOURCE)
    durations: list[float] = []
    for _ in range(n_calls):
        started = time.perf_counter()
        _post_generate(prompt, model_tag=model_tag, host=host, timeout=timeout)
        durations.append(time.perf_counter() - started)
    per_item = sum(durations) / len(durations)
    estimate = ThroughputEstimate(
        model_tag=model_tag,
        n_items=n_items,
        call_seconds=tuple(durations),
        seconds_per_item=per_item,
        estimated_seconds=per_item * n_items,
    )
    _log.info("throughput estimate: %s", estimate.summary())
    return estimate


# --- talking to the model ------------------------------------------------------


def _post_generate(prompt: str, *, model_tag: str, host: str, timeout: float) -> str:
    """POST one prompt to Ollama and return the raw completion text.

    Every failure path raises :class:`SemanticAuditUnavailable` with the host and the remedy in
    the message, because the usual cause is simply that ``ollama serve`` is not running and the
    person reading the traceback should not have to go looking for that.
    """
    url = f"{host}{GENERATE_PATH}"
    payload: dict[str, Any] = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        # Ollama constrains decoding to valid JSON when asked. The defensive parser downstream
        # stays regardless: this flag is honoured by recent servers only, and even when honoured
        # it guarantees JSON, not the keys we asked for.
        "format": "json",
        "options": {"temperature": TEMPERATURE, "seed": SEED, "num_ctx": NUM_CTX},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise SemanticAuditUnavailable(
            f"could not reach Ollama at {url}: {exc}. Start it with `ollama serve` and confirm "
            f"the model is present with `ollama pull {model_tag}`."
        ) from exc
    if response.status_code != 200:
        raise SemanticAuditUnavailable(
            f"Ollama at {url} returned HTTP {response.status_code}: {response.text[:200]}. "
            f"A 404 usually means the model tag is not pulled (`ollama pull {model_tag}`)."
        )
    try:
        body: object = response.json()
    except ValueError as exc:
        raise SemanticAuditUnavailable(
            f"Ollama at {url} returned a non-JSON body: {response.text[:200]}"
        ) from exc
    if not isinstance(body, dict):
        raise SemanticAuditUnavailable(f"Ollama at {url} returned {type(body).__name__}, not an "
                                       "object")
    text = body.get("response")
    if not isinstance(text, str) or not text.strip():
        raise SemanticAuditUnavailable(
            f"Ollama at {url} returned an empty completion for model {model_tag}"
        )
    return text


# --- reading what came back ----------------------------------------------------


def _extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block in ``text``.

    Small local models wrap their answer in markdown fences or a sentence of preamble often
    enough that insisting on a bare object would throw away good findings. Scanning for a balanced
    object handles fences, prose, and trailing commentary alike, with string and escape tracking so
    a brace inside an explanation does not close the object early. If the first candidate never
    balances - a truncated reply - the scan restarts at the next opening brace.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : position + 1]
        start = text.find("{", start + 1)
    raise SemanticAuditParseError(
        f"no balanced JSON object in the model reply: {text[:300]!r}"
    )


def _coerce_confidence(value: object, raw: str) -> float:
    """Read the confidence field as a float in [0, 1], clamping and logging when out of range."""
    complaint = f"confidence is not a number: {value!r} (reply {raw[:200]!r})"
    # bool is a subclass of int, so `True` would otherwise sail through as a confidence of 1.0.
    if isinstance(value, bool) or value is None:
        raise SemanticAuditParseError(complaint)
    if isinstance(value, str):
        try:
            number = float(value.strip().rstrip("%"))
        except ValueError as exc:
            raise SemanticAuditParseError(complaint) from exc
    elif isinstance(value, int | float):
        number = float(value)
    else:
        raise SemanticAuditParseError(complaint)
    clamped = min(1.0, max(0.0, number))
    if clamped != number:
        # Worth seeing rather than silently fixing: a model returning 95 for a 0-1 field has
        # misread the instructions, which casts some doubt on the label it returned with it.
        _log.warning("clamped out-of-range confidence %r to %.2f", number, clamped)
    return clamped


def _parse_response(raw: str, *, model_tag: str) -> SemanticFinding:
    """Turn a raw completion into a validated finding, raising on anything unreadable."""
    blob = _extract_json_object(raw)
    try:
        parsed: object = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise SemanticAuditParseError(
            f"model reply is not valid JSON ({exc}): {blob[:300]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise SemanticAuditParseError(f"expected a JSON object, got {type(parsed).__name__}")

    label = parsed.get("label")
    if not isinstance(label, str) or label.strip() not in LABELS:
        # Not coerced to `consistent`: a label outside the taxonomy means the model did not do the
        # task it was given, and treating that as a pass is exactly the failure this layer exists
        # to catch in other people's code.
        raise SemanticAuditParseError(
            f"label {label!r} is not one of {LABELS} (reply {raw[:200]!r})"
        )
    if "confidence" not in parsed:
        raise SemanticAuditParseError(f"model reply has no confidence field: {blob[:300]!r}")
    if "explanation" not in parsed:
        raise SemanticAuditParseError(f"model reply has no explanation field: {blob[:300]!r}")

    return SemanticFinding(
        label=label.strip(),
        confidence=_coerce_confidence(parsed["confidence"], raw),
        explanation=str(parsed["explanation"]).strip(),
        model_tag=model_tag,
        raw_response=raw,
    )
