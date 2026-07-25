"""Prompt text and label taxonomy for the semantic audit layer.

Kept apart from the code that calls the model because the wording *is* part of the method. A
change to a single sentence here can move every label the layer assigns, so it belongs in one
reviewable place where it shows up plainly in a diff, rather than inlined into a request builder
where nobody looks. Nothing in this module performs I/O.

The four labels live here too. The taxonomy and the text that teaches the taxonomy must not be
allowed to drift apart, and they will if they sit in different files.
"""

from __future__ import annotations

from typing import Final

#: The complete label set. A response naming anything else is a parse failure, not a new category.
#: Order matters: it is the precedence the model is told to apply when more than one label fits,
#: running from the most concrete defect to the default.
LABELS: Final[tuple[str, ...]] = (
    "rationale_implementation_mismatch",
    "unfalsifiable_mechanism",
    "unacknowledged_known_anomaly",
    "consistent",
)

#: Prompt-side truncation limits. A 7B model on 8 GB of VRAM runs a modest context window, and a
#: prompt that overflows it is silently cut from the far end — which would drop the strategy code
#: while keeping the instructions, producing a confident label formed from almost no evidence.
#: Clipping here instead means the truncation is explicit, marked in the text, and visible to the
#: model. Findings on a clipped item are weaker evidence; the marker is what makes that auditable.
MAX_RATIONALE_CHARS: Final[int] = 2_000
MAX_SOURCE_CHARS: Final[int] = 6_000

_CLIP_MARKER: Final[str] = "\n... [truncated for context length; the remainder was not shown]"

INSTRUCTIONS: Final[str] = """\
You are auditing one candidate trading strategy. You are given the author's stated economic
rationale and the source code that implements it. Your only question is whether the rationale is
an honest description of what the code does.

Assign exactly one label from this list.

rationale_implementation_mismatch
    The code does not do what the rationale claims. For example: the rationale describes buying
    recent losers but the code buys recent winners; the rationale names a one-month window and
    the code uses a year; the rationale describes a risk limit the code never applies.

unfalsifiable_mechanism
    The rationale is phrased so that no outcome could contradict it. For example: "the signal
    captures market psychology, which asserts itself over the appropriate horizon"; "prices
    revert when they are stretched and trend when they are not". A rationale that explains a
    profit and a loss equally well belongs here.

unacknowledged_known_anomaly
    The rationale presents a long-documented effect - momentum, low volatility, size, value, or
    short-term reversal - as a novel, proprietary, or newly discovered one, without acknowledging
    that it is already well known. Implementing a known effect is not a defect. Claiming to have
    found it is.

consistent
    None of the above. The rationale describes what the code does. It may well be wrong about the
    market; that is not your concern.

Rules.
1. Judge the code as written. Do not speculate about what the author meant, and do not assume
   anything about code you cannot see.
2. If you are unsure, answer consistent. A false alarm costs more here than a missed defect: an
   auditor that flags everything gets discarded and audits nothing. Report only a defect you can
   point at in the text in front of you.
3. Being simple, unoriginal, or unprofitable is not a defect. Performance is measured elsewhere.
4. If more than one label applies, report the first one that applies in the order listed above.
5. confidence is your own probability that the label is correct, from 0 to 1. When you hesitate,
   lower the confidence rather than changing the label to something you can defend less.

Reply with one JSON object and nothing else, in exactly this shape:
{"label": "<one of the four labels>", "confidence": <number from 0 to 1>, "explanation": "<at \
most 40 words, quoting the specific phrase or line that decided it>"}
"""

# Three worked examples, not four. The prompt has to fit a small context window alongside the
# strategy source, so one label goes without an exemplar; unfalsifiable_mechanism is the one that
# reads most directly off its own definition, so it is the one left out. The `consistent` example
# is not optional - without it the model learns that being shown code means finding fault with it.
FEW_SHOT_EXAMPLES: Final[str] = """\
Here are three worked examples.

--- EXAMPLE 1 ---
RATIONALE:
Stocks that have fallen hardest over the past week tend to bounce as forced selling exhausts
itself, so the portfolio buys the biggest one-week losers.

CODE:
scores = returns_5d(view)
picks = sorted(scores, key=scores.get, reverse=True)[:10]

ANSWER:
{"label": "rationale_implementation_mismatch", "confidence": 0.93, "explanation": "The rationale \
buys the biggest losers; sorting with reverse=True and taking the head selects the ten highest \
five-day returns, i.e. the winners."}

--- EXAMPLE 2 ---
RATIONALE:
We have identified a persistent inefficiency, unique to this framework: names with the strongest
trailing six-month returns keep outperforming. This relationship does not appear in the
literature and forms the core of our edge.

CODE:
scores = total_return(view, lookback=126)
picks = sorted(scores, key=scores.get, reverse=True)[:20]

ANSWER:
{"label": "unacknowledged_known_anomaly", "confidence": 0.88, "explanation": "Twelve-month-scale \
cross-sectional momentum is one of the most documented effects in equities, but the rationale \
claims it is unique to this framework and absent from the literature."}

--- EXAMPLE 3 ---
RATIONALE:
A short moving average above a long one means recent prices are running ahead of the established
level, the conventional definition of an uptrend. The portfolio holds only names in that state.

CODE:
short_avg = mean(closes[-20:])
long_avg = mean(closes[-50:])
if short_avg > long_avg:
    picks.append(symbol)

ANSWER:
{"label": "consistent", "confidence": 0.9, "explanation": "The code computes a 20-day and a \
50-day average and holds names where the short exceeds the long, which is what the rationale \
describes. No novelty is claimed."}
"""

#: Fixed item used by the throughput probe. It is a realistic length rather than a toy string,
#: because an estimate timed on a two-line prompt would understate the real batch by a wide margin
#: and the point of the probe is to report an honest wall-clock figure before committing to a run.
PROBE_RATIONALE: Final[str] = (
    "Names whose price sits above their 200-day average are in an established uptrend, and the "
    "portfolio holds those names in equal weight while staying out of the rest."
)
PROBE_SOURCE: Final[str] = """\
class PriceAboveSma200(Strategy):
    def generate(self, view: MarketView) -> Signal:
        closes = view.closes(lookback=200)
        picks = []
        for symbol in view.symbols:
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < 200:
                continue
            if values[-1] > sum(values) / len(values):
                picks.append(symbol)
        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(information_available_at=latest_visible(view),
                      weights=dict.fromkeys(picks, weight))
"""


def clip(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, leaving a visible marker when anything was removed."""
    if len(text) <= limit:
        return text
    return text[:limit] + _CLIP_MARKER


def build_prompt(rationale: str, source_code: str) -> str:
    """Assemble the full prompt for one strategy.

    Built by concatenation rather than ``str.format``: the few-shot answers are literal JSON, and
    running braces through a format call would either raise or require escaping the examples into
    something that no longer matches what we are asking the model to emit.
    """
    return (
        INSTRUCTIONS
        + "\n"
        + FEW_SHOT_EXAMPLES
        + "\n--- ITEM UNDER AUDIT ---\nRATIONALE:\n"
        + clip(rationale.strip() or "(the author supplied no rationale)", MAX_RATIONALE_CHARS)
        + "\n\nCODE:\n"
        + clip(source_code.strip() or "(no source code was supplied)", MAX_SOURCE_CHARS)
        + "\n\nANSWER:\n"
    )
