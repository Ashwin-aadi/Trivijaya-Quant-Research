from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well relative to the market in recent periods to continue outperforming. This "
        "strategy ranks assets based on their performance and invests in the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < (self._window * len(view.symbols)):
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(
            pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("mom")
        )

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mom_series = closes.sort("mom", descending=True)["mom"].to_list()
        picks: list[str] = [row[0] for row in mom_series[: self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest