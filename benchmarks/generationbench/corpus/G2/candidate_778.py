from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects can arise due to recurring events or economic conditions that "
        "influence stock prices. For instance, certain sectors may experience higher demand or "
        "activity during specific times of the year, leading to higher returns."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = history["symbol"].item()
        closes = history.select(pl.col("adj_close"))

        # Calculate the average close for each day of the year
        yearly_closes = (
            closes.group_by(pl.date.DateChunkedColumn(history["session_date"])).mean().transpose()
        )
        avg_closes = {date_str: float(close) for date_str, close in yearly_closes.rows()}

        today = stamp.toordinal()
        seasonal_index = 365 + (today - (today % 365)) // 30
        seasonally_adj_close = avg_closes.get(date(seasonal_index).toordinal(), history["adj_close"][0])

        if view.as_of.month == 12 and view.as_of.day >= 25:
            # Check for end-of-year trend
            recent_highs = (
                history.select(pl.col("close").alias("high"))
                .groupby("symbol")
                .agg((pl.col("high") / seasonally_adj_close - 1.0).mean().alias("recent_return"))
                .sort("recent_return", descending=True)
            )
            picks = [symbol for symbol, _ in recent_highs.sort("recent_return", descending=True).rows()][:5]
        else:
            picks = []

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