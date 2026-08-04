"""Ask the local model for one candidate strategy and check it is usable before accepting it.

The generator's only job is to produce executable candidates. It does not judge them — that is the
auditor's work, and a generator that filtered on quality would decide the study's outcome before
the measurement began.

**Every attempt is a trial, including the ones that fail.** A candidate that does not parse still
consumed a draw from the search, and the Deflated Sharpe Ratio deflates by the number of draws. The
caller records the outcome against the ledger in `src.audit.stat`; this module reports the outcome
honestly and never silently retries past its bound.

**Seeding.** Each candidate takes ``base_seed + index``. This matters more than it looks: a fixed
seed against a fixed prompt makes the model deterministic, and a batch generated that way is one
strategy repeated N times with a trial counter claiming N distinct draws. That was measured before
the first batch ran - nine consecutive calls at seed 42 returned byte-identical output - and the
per-candidate offset is what prevents it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Final

import requests

from src.audit.semantic import GENERATE_PATH, MODEL_TAG, NUM_CTX, OLLAMA_HOST
from src.common.log import get_logger
from src.generate.prompts import build_prompt, theme_for
from src.generate.tokens import Usage

_log = get_logger(__name__)

#: Sampling temperature. Non-zero because the corpus needs variety across candidates; the seed is
#: what makes each one reproducible.
TEMPERATURE: Final[float] = 0.8

#: Attempts per candidate before giving up and recording a failure. Bounded so one pathological
#: theme cannot stall a batch.
MAX_ATTEMPTS: Final[int] = 3

REQUEST_TIMEOUT_SECONDS: Final[float] = 600.0

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Candidate:
    """One generated strategy and the record of how it was obtained."""

    index: int
    seed: int
    theme: str
    source: str
    class_name: str
    attempts: int
    #: "evaluated" once it parses and conforms; otherwise the reason it did not.
    outcome: str
    #: One entry per draw from the model, in order. A retry consumed a draw exactly as the first
    #: attempt did, so the Deflated Sharpe denominator must count all of them - recording only the
    #: final outcome understates N and weakens the deflation, which is the wrong direction.
    attempt_outcomes: tuple[str, ...] = ()
    #: Tokens spent across every attempt, including the ones that failed. Defaults to zero so that
    #: P1's corpus, which was generated before this was recorded, deserialises without claiming a
    #: cost it never measured. A zero here means "not measured", not "free".
    usage: Usage = Usage()

    @property
    def usable(self) -> bool:
        return self.outcome == "evaluated"


def extract_code(text: str) -> str:
    """Pull Python out of a fenced block, or take the reply whole if it is bare code.

    The prompt asks for no fence and the model usually complies, but not always. Stripping one when
    present is not leniency about correctness — the code still has to parse and conform.
    """
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def _conformance_failure(source: str) -> str | None:
    """Why this source cannot be run as a strategy, or None if it can.

    Checks the contract the engine relies on: a class, a `generate` method, and a rationale for the
    semantic layer to read. A candidate missing any of these cannot be evaluated, and pretending
    otherwise would put an unrunnable file into the corpus and a false "evaluated" into the ledger.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if not classes:
        return "no class defined"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "generate" for n in ast.walk(tree)):
        return "no generate method"
    if not any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rationale" for t in n.targets)
        for n in ast.walk(tree)
    ):
        return "no rationale"

    # A constructor with a required parameter cannot be built by the harness, which instantiates
    # with no arguments. The first run lost 21 candidates to this: the prompt invited scalar
    # settings without saying they needed defaults, and the loader assumed they had them. Checking
    # it here means the mismatch is caught at generation rather than discovered a backtest later.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        positional = [a.arg for a in node.args.args if a.arg != "self"]
        n_defaults = len(node.args.defaults)
        if len(positional) > n_defaults:
            missing = positional[: len(positional) - n_defaults]
            return f"__init__ parameters without defaults: {', '.join(missing)}"
    return None


def _class_name(source: str) -> str:
    """Name of the last class defined, which is the strategy by convention of the prompt."""
    tree = ast.parse(source)
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return names[-1] if names else "Unknown"


def _post(prompt: str, seed: int, *, model_tag: str, host: str) -> tuple[str, Usage]:
    """Issue one call and return its text together with what it cost.

    **The payload is unchanged from the one that generated P1's corpus.** Only the return value
    grew, to carry the token counts Ollama has always sent back and this function used to drop.
    P4 compares paradigms at equal token budget under RULE 11, which is unmeasurable without them.
    Altering the request instead would have made P1's corpus unusable as P4's control arm.
    """
    payload: dict[str, str | bool | dict[str, float | int]] = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": seed, "num_ctx": NUM_CTX},
    }
    response = requests.post(
        f"{host}{GENERATE_PATH}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    body = response.json()
    return str(body.get("response", "")), Usage.from_ollama(body)


def generate_candidate(
    index: int,
    *,
    base_seed: int = 42,
    model_tag: str = MODEL_TAG,
    host: str = OLLAMA_HOST,
) -> Candidate:
    """Produce candidate ``index``, retrying up to ``MAX_ATTEMPTS`` if the output is unusable.

    Returns a Candidate either way. A failure is a result to be recorded, not an exception to be
    swallowed — the caller must count it as a trial.
    """
    theme = theme_for(index)
    prompt = build_prompt(theme)
    last_reason = "no attempt made"
    outcomes: list[str] = []
    spent = Usage()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Offsetting by attempt as well as index keeps a retry from redrawing the same failure.
        seed = base_seed + index + attempt * 10_000
        text, usage = _post(prompt, seed, model_tag=model_tag, host=host)
        # Accumulated before the conformance check, so a failed attempt is still charged for the
        # tokens it burned. Charging only successes would flatter whichever paradigm retries most.
        spent = spent + usage
        source = extract_code(text)
        reason = _conformance_failure(source)
        if reason is None:
            outcomes.append("evaluated")
            return Candidate(
                index=index, seed=seed, theme=theme, source=source,
                class_name=_class_name(source), attempts=attempt, outcome="evaluated",
                attempt_outcomes=tuple(outcomes), usage=spent,
            )
        last_reason = reason
        outcomes.append("syntax_error" if "syntax" in reason else "runtime_error")
        _log.warning("candidate %d attempt %d unusable: %s", index, attempt, reason)

    return Candidate(
        index=index, seed=base_seed + index, theme=theme, source="",
        class_name=f"Failed{index}", attempts=MAX_ATTEMPTS,
        outcome="syntax_error" if "syntax" in last_reason else "runtime_error",
        attempt_outcomes=tuple(outcomes), usage=spent,
    )
