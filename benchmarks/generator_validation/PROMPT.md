# The prompt, and how it differs from P1's

The text issued to every frontier arm is [`PROMPT.txt`](PROMPT.txt), pasted verbatim. It is held in
its own file because the wording determines what the corpus contains, and any comparison across
arms is only meaningful if the digest matches.

| | Digest (SHA256) | Length |
|---|---|---|
| P1 frozen prompt, `src/generate/prompts.py` | `f307433c7bda8595d52432b3bcb4f723663bfe706112a41f84e4beacfbde9934` | 5,191 chars |
| Addendum prompt, `PROMPT.txt` | `89c6013cc46fbd4e63fbdcd38b1e7b871d0bb637f90a91383988797b33f607d5` | 5,567 chars |

**The digests differ, and that is a disclosed feature of the design rather than an oversight.**

## What is identical

The interface contract, verbatim: the `Strategy` subclass shape, every `MarketView` accessor, the
`information_available_at` requirement and its warning, the polars 1.43 error list, the correct-usage
examples, the plain-Python advice, the requirements block, the complete worked `Breakout20d`
example, and the closing note that `closes()` is wide while `history()` is long.

**It still says nothing about leakage, lookahead or point-in-time discipline.** Warning a generator
away from the failure modes being measured would suppress the population the study exists to
characterise, and the resulting rejection rate would describe the prompt rather than the model. That
reasoning is P1's and it carries over unchanged.

## What differs, and why

1. **Five strategies in one Markdown file, rather than one Python file per call.** This is the
   realistic-usage condition under study, ruled by the PI: a working researcher asks once and gets a
   file back. It is not a defect to be corrected — it is the exposure being measured.
2. **Five named themes in one request**, rather than one theme drawn by index from the cycle of
   twelve. The five are the first five of P1's `THEMES`, in P1's order, so the theme mix overlaps
   the reference corpus rather than being freshly invented.
3. **An output-format instruction** — heading, fenced block, distinct class names — so the file can
   be parsed mechanically. P1 asked for bare source with no fence because it consumed one file per
   call.

## The consequence for interpretation

Draws within one response are **not independent**. The model sees its own preceding strategies and
can be expected to differentiate them deliberately. This inflates measured diversity and depresses
measured duplication relative to M₀, whose draws were independent API calls.

**Hypothesis H4 is affected in the direction opposite to its prediction**, which is worth stating
plainly: if near-duplicates appear *despite* the batching advantage, that is stronger evidence for
mode collapse than the same finding from independent draws. If they do not appear, the batching
alone may explain it, and H4 must be reported as inconclusive rather than falsified.

## Collection conditions

Fixed before generation, and part of the record:

- A new conversation per request, with memory, personalisation, custom instructions and project or
  workspace context **disabled**.
- The raw response saved verbatim and hashed before any parsing.
- No response regenerated, edited, retried or discarded — including refusals, truncation and
  malformed output. A failed response is a datum.
- Product name and access date recorded per request. This measures three products on specific
  dates, not three laboratories.
