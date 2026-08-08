from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "The systematic equity trading strategy focuses on low-volatility tilting, exploiting the empirical observation that low-volatility stocks tend to outperform high-volatility stocks over time. By systematically selecting stocks with lower historical volatility and maintaining exposure to the broader market, this approach aims to capture persistent risk-adjusted returns while minimizing exposure to extreme market fluctuations."
    )

    def __init__(self, lookback_window: int = 12 * 5, top_quintile_size: int = 20) -> None:
        self._lookback_window = lookback_window
        self._top_quintile_size = top_quintile_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["symbol"])
        )

        # Compute trailing 12-month volatility
        volatilities = (
            history.groupby("symbol").agg(
                (pl.col("returns") ** 2.0).mean().alias("variance"),
                ((pl.col("returns") - pl.col("returns").mean()).pow(2)).sum().alias("total_variance")
            )
            .with_columns(
                (pl.col("total_variance") / self._lookback_window).alias("volatility")
            )
        )

        # Rank by volatility
        ranked = volatilities.sort(by="volatility", descending=False)

        if ranked.height < self._top_quintile_size:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = [symbol for symbol in ranked.slice(0, self._top_quintile_size)["symbol"].to_list()]
        weight_per_stock = 1.0 / len(top_stocks)

        return Signal(
            information_available_at=stamp,
            weights={
                s: weight_per_stock
                for s in top_stocks
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest