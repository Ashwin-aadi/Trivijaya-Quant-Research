from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Historical data in Indian markets suggests that certain seasons or months exhibit "
        "repetitive patterns of higher returns. This strategy aims to capitalize on these trends."
    )

    def __init__(self, lookback_period: int = 5) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)

        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        symbol = history["symbol"].item()
        closes = [float(v) for v in history.select("adj_close").to_pandas().iloc[:, 0].dropna()]

        month_of_year = stamp.month
        historical_closes_by_month = (
            view.history(lookback=self._lookback_period * 12).select("symbol", "session_date", "adj_close")
            .with_columns(pl.col("session_date").dt.replace_time_zone("UTC").dt.strftime("%Y-%m"))
            .group_by("symbol", pl.col("session_date").dt.month())
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                (pl.col("session_date")).alias("date")
            )
            .sort("symbol", "date").collect()
        )

        returns_by_month = historical_closes_by_month.filter(pl.col("date").dt.month() == month_of_year)
        if returns_by_month.height < 12:
            return Signal(information_available_at=stamp, weights={})

        mean_return = float(returns_by_month.select(pl.col("return")).mean().item())
        recent_close = view.latest_close()[symbol]

        if (recent_close / recent_close * (1 + mean_return)) > recent_close:
            weight = 1.0
        else:
            weight = 0.0

        return Signal(
            information_available_at=stamp, weights={symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest