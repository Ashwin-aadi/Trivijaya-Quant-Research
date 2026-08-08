from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion is a common market phenomenon where prices tend to move back towards "
        "historical means after extreme deviations. By identifying symbols that have deviated"
        " significantly from their mean price over the past 20 days, we can anticipate a potential"
        " reversion and profit from such movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").mean().over("symbol")).alias(
                "reversion_score"
            ),
        ).collect()

        # Filter out symbols where the reversion score is within normal range
        mean_price = mean_price.filter(
            (pl.col("reversion_score") < 0.95) & (pl.col("reversion_score") > 1.05)
        )

        if mean_price.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Sort symbols by reversion score in descending order
        sorted_symbols = mean_price.sort("reversion_score", descending=True)

        top_symbols = [row["symbol"] for row in sorted_symbols.rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest