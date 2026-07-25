"""Buy the names that opened the most recent completed session well below the previous close."""

from __future__ import annotations

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class GapFade(Strategy):
    """Fades large downward overnight gaps, using two completed sessions."""

    rationale = (
        "A large downward overnight gap is often a liquidity event rather than a revaluation: "
        "orders accumulated while the market was shut clear against a thin opening book, and part "
        "of the move is that impact. Fading it means buying the names that gapped down hardest "
        "and holding while the price works back. It fails whenever the gap was information."
    )

    def __init__(self, threshold: float = 0.02, holdings: int = 5) -> None:
        if threshold <= 0:
            raise ValueError("the gap threshold must be positive")
        self._threshold = threshold
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=2)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(
                ["adj_open", "adj_close"]
            ).sort("session_date")
            if rows.height < 2:
                continue
            # Both inputs are completed sessions: the opening print of the most recent visible
            # session, and the close of the session before it. The session being traded, whose
            # open is where this order fills, contributes nothing to the score.
            opening = float(rows["adj_open"].to_list()[-1])
            prior_close = float(rows["adj_close"].to_list()[-2])
            if prior_close <= 0:
                continue
            gap = opening / prior_close - 1.0
            if gap <= -self._threshold:
                scores[symbol] = gap
        # Smallest-first, so the most severe gaps are the ones bought.
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings, largest=False)),
        )
