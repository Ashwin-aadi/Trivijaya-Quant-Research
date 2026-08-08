from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices tend to revert back to their historical averages over short periods. "
        "This strategy identifies stocks with significant negative deviations from a 10-day moving average and initiates long positions in the top 25% of such stocks, exiting if daily returns exceed the moving average or after a fixed holding period."
    )

    def __init__(self, window: int = 10, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        closes = view.closes(lookback=self._window)
        prev_closes = closes.drop("session_date").shift(1).fill_null(strategy="forward")
        returns = (closes["adj_close"] - prev_closes) / prev_closes

        # Calculate 10-day moving average of daily returns
        avg_return = history.select(
            pl.col("symbol"),
            (pl.col("close") - pl.col("open")).mean().over(pl.date_range(history["session_date"].min(), history["session_date"], interval="1 day")) / history["open"].shift(1).fill_null(strategy="forward").mean().over(pl.date_range(history["session_date"].min(), history["session_date"], interval="1 day")),
        ).with_columns(
            (pl.col("close") - pl.col("open")).rank(method="dense", descending=True).alias("rank")
        )
        
        # Identify stocks with negative 10-day average returns
        avg_return_filter = avg_return.filter(pl.col("rank") <= self._window)
        picks: list[str] = [symbol for symbol in view.symbols if symbol in avg_return_filter.columns and float(avg_return_filter.select(pl.col(symbol).max().alias("max_val")).collect()["max_val"][0]) < 0]

        # Rank stocks based on deviation from the mean
        dev_from_mean = (returns - returns.mean()).drop_nulls().abs()
        ranked_dev = dev_from_mean.sort("session_date").select(
            pl.col("symbol"),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).rank(method="dense", descending=True).alias("rank")
        )

        # Select top N stocks
        top_stocks = ranked_dev.top(self._top_n, "rank")

        picks = [symbol for symbol in top_stocks.select("symbol").to_dict(as_series=False)["symbol"]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest