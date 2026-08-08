from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in the recent past to continue outperforming. This strategy "
        "aims to identify such stocks and allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol
        history_with_returns = (
            history
            .with_column(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .select([pl.col("symbol"), "return"])
        )

        # Rank symbols by return
        ranked_returns = (
            history_with_returns
            .group_by("symbol")
            .agg(
                pl.col("return").mean().alias("avg_return"),
                pl.col("return").rank(method="ordinal", descending=True).alias("rank"),
            )
            .sort("rank")
        )

        # Select top performing symbols
        top_performers = ranked_returns.select(["symbol"]).head(5)

        if top_performers.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest