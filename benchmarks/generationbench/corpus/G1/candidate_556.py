from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price levels that revert to the mean after a significant move are often good "
        "entry points. This strategy identifies symbols where the current price has moved "
        "significantly from its 20-day average and is now close to that average, indicating "
        "a potential reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.groupby(pl.col("symbol")).agg(
                (pl.col("adj_close").mean()).alias("20d_mean")
            )
            .select(
                pl.col("symbol"), "20d_mean"
            )  # Ensure these columns are selected for later joining
            .with_columns(
                (pl.col("adj_close") - pl.col("20d_mean")).alias("deviation"),
                (pl.col("adj_close") / pl.col("20d_mean")).alias("price_ratio"),
            )
        )

        # Join mean_close with the latest closes to get recent prices
        joined = (
            closes.lazy()
            .join(mean_close, on="symbol", how="inner")
            .select(
                pl.col("session_date"), "symbol", "adj_close", "20d_mean", "deviation", "price_ratio"
            )
            .sort("session_date", descending=True)
        ).collect()

        # Get the latest deviation and price ratio
        recent_data = joined.to_pandas().iloc[0]
        symbol = recent_data["symbol"]
        deviation = recent_data["deviation"]
        price_ratio = recent_data["price_ratio"]

        if abs(deviation) > 1.5 and price_ratio < 0.95:
            return Signal(
                information_available_at=stamp,
                weights={symbol: 1.0},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest