from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in the recent past to continue outperforming them. This strategy "
        "identifies such stocks and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not closes.columns:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns over the window period
        returns = (
            (closes / closes.shift(1) - 1.0).drop_nulls()
        ).rename("returns")

        # Get the latest close prices to rank symbols by recent performance
        latest_closes = view.latest_close()
        ranked_symbols = (
            returns.join(latest_closes, on="symbol")
            .sort(by="returns", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate capital based on the top N performing symbols
        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest