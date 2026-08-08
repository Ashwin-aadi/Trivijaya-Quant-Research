from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks which have performed well relative to "
        "the market over the recent past are likely to continue performing well in the near future. "
        "This strategy exploits this phenomenon by buying winners and selling losers."
    )

    def __init__(self, window: int = 20, top_n_winners: int = 5, bottom_n_losers: int = 5) -> None:
        self._window = window
        self._top_n_winners = top_n_winners
        self._bottom_n_losers = bottom_n_losers

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        winners: list[str] = []
        losers: list[str] = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            price_changes = [
                float(v) for v in (history[symbol]["close"] / history[symbol]["close"].shift(1) - 1.0).to_list()
            ]
            if len(price_changes) < self._window:
                continue

            rank = pl.Series(price_changes).rank(method="ordinal", descending=True)
            winner_rank = int(len(price_changes) * (self._top_n_winners / 100))
            loser_rank = int(len(price_changes) * (self._bottom_n_losers / 100))

            if rank[0] <= winner_rank:
                winners.append(symbol)
            elif rank[0] >= len(price_changes) - loser_rank:
                losers.append(symbol)

        if not winners or not losers:
            return Signal(information_available_at=stamp, weights={})

        weight_winners = 1.0 / len(winners)
        weight_losers = -1.0 / len(losers)

        return Signal(
            information_available_at=stamp,
            weights={
                s: weight_winners for s in winners
            } | {
                s: weight_losers for s in losers
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest