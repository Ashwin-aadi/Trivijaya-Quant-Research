from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalValueStrategy(Strategy):
    rationale = (
        "This strategy exploits seasonality in the Indian equity market by identifying undervalued "
        "stocks before key calendar events. It leverages historical patterns and risk management to "
        "achieve potentially higher returns."
    )

    def __init__(self, window: int = 30, lookback: int = 252, top_n: int = 20) -> None:
        self._window = window
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate historical returns and volatility
        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = view.closes(lookback=self._window)
        means = (
            history.select(["session_date", "adj_close"])
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("mean_adj_close"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
        )

        volatilities = (
            means.select(["symbol", "return"])
            .group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
        )

        # Rank symbols based on value score and seasonal factor
        value_scores = (
            closes.join(means, on="symbol", how="left")
            .join(volatilities, on="symbol", how="left")
            .sort(
                "mean_adj_close",
                descending=True,
            )
            .with_columns(
                (pl.col("return") - pl.col("mean_adj_close")).alias("historical_return"),
                (1 / pl.col("volatility")).alias("inverse_volatility"),
            )
            .select(["symbol", "historical_return", "inverse_volatility"])
        )

        seasonal_factors = {
            symbol: 1.0 if month in [12, 1, 2] else 0.5
            for symbol, month in zip(symbols, _get_month_of_year(view.as_of))
        }

        combined_scores = value_scores.with_columns(
            (pl.col("historical_return") + pl.col("inverse_volatility") * seasonal_factors[pl.col("symbol")]).alias("combined_score")
        )

        top_symbols = (
            combined_scores.sort("combined_score", descending=True)
            .select(["symbol"])
            .head(self._top_n)["symbol"]
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in top_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _get_month_of_year(date_obj: date) -> int:
    return date_obj.month