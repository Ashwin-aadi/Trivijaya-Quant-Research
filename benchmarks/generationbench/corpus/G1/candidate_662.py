from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following is a strategy that follows the trend of an asset "
        "while adjusting position size based on historical volatility. High volatility periods "
        "reduce the exposure to protect against significant drawdowns."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            pl.DataFrame(
                {
                    "symbol": closes["symbol"],
                    "volatility": (closes["adj_close"].rolling_std(window=self._vol_window) / closes["adj_close"])
                                  .drop_nulls()
                                  .to_list(),
                }
            )
            .with_columns((pl.col("volatility") * 10).round(2).alias("scaled_volatility"))
        )

        trends = (
            volatility.sort("symbol")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().sum()
                / pl.col("volatility").mean()
                * 0.5
            )
            .collect()["scaled_volatility"]
        ).to_list()

        picks = []
        for symbol, trend in zip(volatility["symbol"].drop_nulls().to_list(), trends):
            if float(trend) > 1:
                picks.append(symbol)

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