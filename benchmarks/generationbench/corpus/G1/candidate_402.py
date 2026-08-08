from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects the top performing stocks relative to the NIFTY 100 index "
        "based on their adjusted closing prices over a lookback period. It assumes that "
        "stocks that have performed well in relation to the broader market may continue "
        "to outperform."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes(lookback=self._window).select(
            pl.all().exclude("session_date")
        )
        individual_closes = history.select(["symbol", "adj_close"])
        if nifty_closes.height < self._window or individual_closes.height < 20:
            return Signal(information_available_at=stamp, weights={})

        nifty_avg = (
            nifty_closes.mean().to_dict()["adj_close"]
        )  # Calculate average of NIFTY closes
        symbols = [
            row["symbol"]
            for row in individual_closes.sort("symbol").select(
                pl.col("symbol")
                .zip_with(pl.col("adj_close") / nifty_avg, pl.mul)
                .rank(method="dense", descending=True)
                .filter(pl.col(0) <= self._window)
                .to_dicts()
            )
        ]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest