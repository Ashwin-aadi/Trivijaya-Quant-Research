from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers over the last period are more likely to "
        "continue this outperformance in the near future. This strategy aims to identify such "
        "stocks and allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        returns = (
            history.select(pl.col("adj_close").to_list())
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .select(["symbol", "session_date", "r"])
        )

        # Calculate relative strength
        rs = (
            returns.groupby("symbol")
            .agg(pl.col("r").mean().alias("avg_return"))
            .with_columns(
                (pl.col("avg_return") / pl.col("avg_return").max() * 100).alias("rs")
            )
        )

        # Get the top performers
        top_performers = rs.sort("rs", descending=True).select("symbol").head(5)

        if not top_performers.height:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={row[0]: weight for row in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest