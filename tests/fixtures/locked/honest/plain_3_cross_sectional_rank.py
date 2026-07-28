"""A plain cross-sectional momentum rank: score every visible name, hold the top scorers.

Negative control: the per-symbol score is computed with a ``group_by("symbol")`` aggregation over
``view.history(lookback)``, itself already bounded to sessions strictly before the decision date.
Sorting the whole cross-section by that score and keeping the head is an ordinary ranking
operation, not a lookahead device.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


def _decision_stamp(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class CrossSectionalMomentumRank(Strategy):
    """Rank the visible universe by trailing return and hold the top decile equally weighted."""

    rationale = (
        "Cross-sectional momentum ranks every name in the investable universe by trailing return "
        "over a common window and holds the strongest relative performers, rather than judging "
        "any one name in isolation. This is the standard construction behind decades of momentum "
        "research, offered here as a plain baseline with no adjustment for costs or crowding."
    )

    def __init__(self, lookback: int = 21, top_n: int = 6) -> None:
        self.lookback = lookback
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        decision_date = _decision_stamp(view)
        frame = view.history(self.lookback)
        if frame.is_empty():
            return Signal(information_available_at=decision_date, weights={})
        per_symbol_return = (
            frame.sort(["symbol", "session_date"])
            .group_by("symbol", maintain_order=True)
            .agg(
                (pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias(
                    "trailing_return"
                )
            )
            .filter(pl.col("symbol").is_in(list(view.symbols)))
        )
        if per_symbol_return.is_empty():
            return Signal(information_available_at=decision_date, weights={})
        ranked_frame = per_symbol_return.sort("trailing_return", descending=True).head(self.top_n)
        symbols = ranked_frame["symbol"].to_list()
        weights: dict[str, float] = {}
        if symbols:
            share = 1.0 / len(symbols)
            for symbol in symbols:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
