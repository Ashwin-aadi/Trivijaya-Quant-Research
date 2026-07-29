from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Stock markets exhibit seasonality due to various economic and corporate events. "
        "This strategy focuses on buying stocks during periods of historical performance."
    )

    def __init__(self, window: int = 20, lookback_periods: int = 12) -> None:
        self._window = window
        self._lookback_periods = lookback_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_periods)
        if history.height < self._lookback_periods * 20 + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate historical returns
        returns: pl.DataFrame = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
        )

        # Filter for the top-performing symbols
        top_symbols = returns.head(self._lookback_periods)["symbol"].to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest