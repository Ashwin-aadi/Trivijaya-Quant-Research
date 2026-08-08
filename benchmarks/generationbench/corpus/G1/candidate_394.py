from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit seasonality where their performance "
        "varies significantly throughout the year. This strategy aims to capitalize on such "
        "seasonal trends by identifying and investing in symbols that have historically performed "
        "well during specific months."
    )

    def __init__(self, lookback_years: int = 3) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        seasonal_trends: dict[str, float] = {}

        for symbol in symbols:
            # Calculate the monthly returns
            monthly_returns = (
                history.select(
                    pl.col("session_date").dt.month_name(),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                )
                .group_by("month")
                .agg(pl.col("return").mean().alias("avg_return"))
                .sort("avg_return", descending=True)
            )

            # Extract the top month(s) with highest returns
            if not monthly_returns.is_empty():
                top_month = monthly_returns.head(1)["avg_return"].to_list()[0]
                seasonal_trends[symbol] = float(top_month)

        if not seasonal_trends:
            return Signal(information_available_at=stamp, weights={})

        # Normalize the trends to sum up to 1
        total_trend = sum(seasonal_trends.values())
        seasonal_weights = {symbol: trend / total_trend for symbol, trend in seasonal_trends.items()}

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in seasonal_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest