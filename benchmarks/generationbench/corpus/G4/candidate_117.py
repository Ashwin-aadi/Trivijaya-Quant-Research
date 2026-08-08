from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical observation that low-volatility stocks tend to outperform high-volatility "
        "stocks over long periods due to risk compensation and market inefficiencies. By tilting portfolios towards "
        "low volatility, we aim to capture superior risk-adjusted returns."
    )

    def __init__(self, window: int = 250, top_n_percent_short: float = 0.2, bottom_n_percent_long: float = 0.4) -> None:
        self._window = window
        self._top_n_percent_short = top_n_percent_short
        self._bottom_n_percent_long = bottom_n_percent_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 250-day historical volatility for each stock
        prices = history.select(
            pl.col("adj_close").alias("price")
        )
        volatilities = prices.groupby("symbol").agg(
            (pl.col("price").std() / pl.col("price").mean()).alias("volatility")
        )

        # Sort by volatility and select top n% for short positions, bottom n% for long positions
        volatilities_sorted = volatilities.sort("volatility").to_pandas()
        symbol_list = list(volatilities_sorted["symbol"].values)
        short_symbols = symbol_list[-int(len(symbol_list) * self._top_n_percent_short):]
        long_symbols = symbol_list[:int(len(symbol_list) * self._bottom_n_percent_long)]

        # Determine weights for selected symbols
        if not long_symbols or not short_symbols:
            return Signal(information_available_at=stamp, weights={})

        long_weight = 1.0 / len(long_symbols)
        short_weight = -1.0 / len(short_symbols)

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: long_weight for symbol in long_symbols
            },
            short_weights={
                symbol: short_weight for symbol in short_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest