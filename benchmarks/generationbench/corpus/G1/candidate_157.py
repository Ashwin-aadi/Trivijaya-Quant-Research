from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which are far from their mean will "
        "tend to revert towards the mean in the short term. This strategy aims to "
        "profit from such reversions by betting on underperforming stocks."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).select(pl.col("mean").item()).to_list()[0]

        underperformers: list[str] = []
        for symbol in view.symbols:
            if symbol not in latest_closes.keys():
                continue
            adj_close = float(latest_closes[symbol])
            if adj_close < mean_close * 0.95:
                underperformers.append(symbol)

        if not underperformers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(underperformers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in underperformers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).select(pl.col("session_date").item()).to_list()[0]
    assert isinstance(newest, date)
    return newest