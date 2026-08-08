from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities that have outperformed their peers over a given period are likely to "
        "continue to do so due to the self-fulfilling nature of market sentiment. This strategy "
        "identifies and invests in such securities by comparing recent returns against the broader"
        " market index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = view.closes(lookback=self._window)

        # Calculate returns for NIFTY 100 constituents
        price_changes = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.sum("return").alias("total_return"))
        )

        # Calculate returns for the broad market index (NIFTY 100)
        nifty100_returns = (
            history.group_by("session_date")
            .agg(pl.col("adj_close").mean().alias("market_avg_close"))
            .join(nifty100_closes, on="session_date", how="left")
            .with_columns(
                (pl.col("adj_close") / pl.col("market_avg_close").shift(self._window) - 1.0).alias("nifty100_return")
            )
        )

        # Calculate the relative strength by comparing each stock's return to the market
        relative_strengths = (
            nifty100_returns.join(price_changes, on="symbol", how="left").with_columns(
                (pl.col("total_return") / pl.col("nifty100_return")).alias("relative_strength")
            ).sort("relative_strength", descending=True)
        )

        # Select the top N stocks based on relative strength
        top_stocks = relative_strengths.head(self._window)["symbol"].to_list()
        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={stock: weight for stock in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest