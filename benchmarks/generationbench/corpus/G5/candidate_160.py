from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels revert to the mean over time. This strategy identifies symbols whose "
        "prices have deviated significantly from their trailing 20-day average and bets on a "
        "reversion towards that average."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
            .agg((pl.col("adj_close").mean()).alias("trailing_mean"))
            .with_columns(
                (pl.col("close") / pl.col("trailing_mean") - 1.0).alias("deviation")
            )
            .sort("deviation", descending=True)
        )

        if mean_close.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [row["symbol"] for row in mean_close.to_dicts()[:5]]
        weight = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest