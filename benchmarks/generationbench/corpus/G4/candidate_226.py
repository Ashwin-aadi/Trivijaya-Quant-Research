from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityIT(Strategy):
    rationale = (
        "Exploiting historical seasonality in the IT sector of Indian equities, where certain months exhibit higher returns due to post-election rallies and end-of-year bonuses. This strategy focuses on identifying and capitalizing on these predictable trends."
    )

    def __init__(self, window: int = 10, lookback_months: int = 36) -> None:
        self._window = window
        self._lookback_months = lookback_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_months)
        if history.is_empty() or history.height < self._window * 20:
            return Signal(information_available_at=stamp, weights={})

        it_sectors = ["INFY", "TCS", "WIPRO", "HCLTECH", "HAVELLS"]
        filtered_history = history.filter(pl.col("symbol").is_in(it_sectors))
        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (
            filtered_history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .sort("symbol", descending=False)
            .collect()
        )

        avg_returns = daily_returns.group_by("symbol").agg(
            pl.col("return").mean().alias("avg_return")
        ).with_columns(
            (pl.col("avg_return") / pl.col("avg_return").max()).alias("rank")
        )

        top_stocks = (
            avg_returns.sort("rank", descending=True)
            .head(30)
            .select(["symbol"])
            .to_dict(as_series=False)["symbol"]
        )
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight_per_stock = 1.0 / len(top_stocks)
        weights = {s: weight_per_stock for s in top_stocks}

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol, weight in weights.items()
                if view.closes().get_column(symbol).is_not_null().sum() >= self._window * 20
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest