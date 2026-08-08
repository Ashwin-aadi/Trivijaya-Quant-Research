from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Historical data often shows that certain stocks perform better during specific times "
        "of the year. This strategy aims to identify such seasonal trends and allocate capital "
        "accordingly."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonality = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            daily_returns = (
                (history.select(pl.col("adj_close")) / history.shift(1).select(pl.col("adj_close"))
                 - 1.0).with_column(pl.col("session_date").alias("date"))
            )
            daily_returns = daily_returns.filter(pl.col("symbol") == symbol)
            daily_returns = daily_returns.sort("date")
            avg_return_by_month = (
                daily_returns.groupby(pl.col("date").dt.month()).agg(
                    pl.col("close").mean().alias("avg_return")
                ).collect()
            )
            seasonality[symbol] = {
                month: float(avg_return) for month, avg_return in
                zip(avg_return_by_month["month"].to_list(), avg_return_by_month["avg_return"].to_list())
            }

        top_symbols = sorted(seasonality.keys(), key=lambda x: sum(
            [seasonality[x][m] for m in range(12)]), reverse=True)[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().to_list()[0]
    assert isinstance(newest, date)
    return newest