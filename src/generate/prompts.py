"""The prompt sent to the local model to obtain one candidate strategy.

Held in its own module because it is a reproducibility artifact. The wording determines what the
corpus contains, so a change to it changes every downstream number — rejection rate, class
distribution, deflated Sharpe — and any comparison across runs is only meaningful if the prompt
hash matches. `prompt_digest` exists so the run manifest can record which wording produced a corpus.

**On what this prompt does not say.** It does not mention leakage, lookahead, point-in-time
discipline, or any defect the auditor looks for. Warning the generator away from the failure modes
being measured would suppress exactly the population the study exists to characterise, and the
resulting rejection rate would describe the prompt rather than the model. The interface contract is
stated plainly and nothing more; whatever the model then does with it is the finding.
"""

from __future__ import annotations

import hashlib
from typing import Final

#: Themes cycled across candidates so the corpus spans more than one idea. The generator picks by
#: index, deterministically, so the corpus is reproducible from the seed alone.
THEMES: Final[tuple[str, ...]] = (
    "cross-sectional momentum",
    "short-horizon mean reversion",
    "volatility-scaled trend following",
    "liquidity-screened equal weighting",
    "price-level reversion against a trailing reference",
    "breakout continuation",
    "relative strength against the broad universe",
    "dispersion or range compression",
    "volume-confirmed directional moves",
    "low-volatility tilting",
    "seasonality or calendar effects",
    "a composite of two weakly related characteristics",
)

INTERFACE: Final[str] = '''\
You are a quantitative researcher proposing a systematic equity strategy for the Indian market
(NIFTY 100 constituents, daily bars, 2020-2024).

Write a Python class conforming exactly to this interface:

    from __future__ import annotations
    from datetime import date
    import polars as pl
    from src.backtest.strategy import MarketView, Signal, Strategy

    class MyStrategy(Strategy):
        rationale = "..."        # 3-4 sentences of economic reasoning
        def __init__(self, ...) -> None: ...
        def generate(self, view: MarketView) -> Signal: ...

MarketView gives you these accessors and no others:
    view.symbols                 -> tuple[str, ...]   the investable universe on this date
    view.history(lookback=None)  -> pl.DataFrame      columns: symbol, session_date, open, high,
                                                      low, close, adj_close, volume
    view.closes(lookback=None)   -> pl.DataFrame      wide: session_date x symbol of adj_close
    view.latest_close()          -> dict[str, float]  most recent close per symbol
    view.as_of                   -> date              the session being decided for

Signal(information_available_at=<date>, weights={symbol: fraction}).
Weights are fractions of portfolio value and need not sum to one; the remainder is cash.

Requirements:
- Use polars as pl. Python 3.11. Type hints on every function. No function over 50 lines.
- Handle the case where history is too short to compute your signal: return an empty Signal.
- Deliver one complete, executable file: imports, the class, and any helper functions.

Propose a strategy based on {theme}.

Output only the Python file. No commentary before or after it, no markdown fence.
'''


def build_prompt(theme: str) -> str:
    """The full prompt for one candidate, on a given theme.

    Substitution is by replace rather than `str.format`: the interface text contains literal braces
    (`{symbol: fraction}`) that `format` would try to interpolate.
    """
    return INTERFACE.replace("{theme}", theme)


def theme_for(index: int) -> str:
    """The theme assigned to candidate ``index``, cycled deterministically."""
    return THEMES[index % len(THEMES)]


def prompt_digest() -> str:
    """SHA256 of the interface text, recorded in the run manifest.

    Two corpora are only comparable if this matches. A reworded prompt is a different experiment,
    and without the digest that difference is invisible six months later.
    """
    return hashlib.sha256(INTERFACE.encode("utf-8")).hexdigest()
