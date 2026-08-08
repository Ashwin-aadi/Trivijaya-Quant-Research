from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain times of the year may see higher returns due to seasonal factors such as "
        "holiday spending, earnings releases, or weather patterns affecting industries. "
        "By identifying these seasonal trends and timing trades accordingly, we can potentially "
        "capitalize on predictable market movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols not in the current period
        symbols_in_period = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols_in_period) < 3:  # Ensure we have enough data points to consider
            return Signal(information_available_at=stamp, weights={})

        # Compute seasonality factor by comparing the current close to historical closes
        seasonal_factors = {}
        for symbol in symbols_in_period:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol)
            ).sort("session_date")
            latest_close = float(view.latest_close()[symbol])
            average_seasonal_factor = (
                symbol_history.with_columns(
                    (pl.col(symbol) / pl.col(symbol).shift(1)).alias("ratio")
                )
                .select(pl.col("ratio").mean())
                .get(0)
            )

            seasonal_factors[symbol] = latest_close / average_seasonal_factor

        # Rank symbols by their seasonality factor
        ranked_symbols = [
            (symbol, factor)
            for symbol, factor in sorted(
                seasonal_factors.items(), key=lambda x: x[1], reverse=True
            )
        ]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s[0] for s in ranked_symbols[:3]]
        weight_per_symbol = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight_per_symbol for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest