from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumStrategy(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks that have outperformed or underperformed relative to their historical performance. "
        "Positive momentum indicates past outperformers, while negative momentum suggests underperformers. Long positions are taken in top-ranked momentum stocks, and short positions in bottom-ranked ones."
    )

    def __init__(self, lookback: int = 252) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = set(closes.columns).intersection(history["symbol"].to_list())

        # Filter data to include only common symbols
        filtered_history = history.select(["session_date", "symbol"]).filter(
            pl.col("symbol").is_in(symbols)
        )
        adjusted_closes = closes.select([pl.col(sym) for sym in symbols])

        # Calculate 1-year returns using simple moving average of close prices
        sma_close = (
            filtered_history.lazy()
            .join(adjusted_closes, on="symbol", how="inner")
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").shift(-self._lookback) / pl.col("adj_close") - 1.0)
                .alias("returns")
            )
            .collect()
        )

        # Rank stocks based on their returns
        ranked_stocks = (
            sma_close.with_columns(pl.col("returns").rank(descending=True, method="dense").alias("rank"))
            .sort("rank")
            .select(["symbol", "rank"])
        )

        # Select top and bottom 30-50 stocks based on their ranks
        num_stocks = min(30, len(ranked_stocks))
        long_symbols = ranked_stocks.head(num_stocks)["symbol"].to_list()
        short_symbols = ranked_stocks.tail(num_stocks)["symbol"].to_list()

        # Assign equal weight to each stock in the respective lists
        weights = {s: 1.0 / num_stocks for s in long_symbols}
        if len(short_symbols) > 0:
            weights.update({s: -1.0 / num_stocks for s in short_symbols})

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest