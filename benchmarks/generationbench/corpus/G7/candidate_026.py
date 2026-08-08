from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends adjusted for volatility by calculating the 20-day "
        "exponentially weighted moving average (EWMA) of daily closing prices and scaling it "
        "by a 10-day standard deviation of daily price returns. This approach aims to capture "
        "the direction of the trend while mitigating risk through volatility adjustment."
    )

    def __init__(self, ewma_window: int = 20, std_window: int = 10, positions: int = 5) -> None:
        self._ewma_window = ewma_window
        self._std_window = std_window
        self._positions = positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._ewma_window + self._std_window)

        if history.height < self._ewma_window + self._std_window:
            return Signal(information_available_at=stamp, weights={})

        ewma_column = f"ewma_{self._ewma_window}"
        std_column = f"std_{self._std_window}"

        # Calculate 20-day EWMA of daily closing prices
        ewma = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").rolling_mean(self._ewma_window)).alias(ewma_column)
            )
            .collect()
        )

        # Calculate 10-day std of daily price returns
        returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("returns")
        std_history = (
            history.lazy()
            .with_columns(returns)
            .group_by("symbol")
            .agg(
                (pl.col("returns").rolling_std(self._std_window)).alias(std_column)
            )
            .collect()
        )

        combined = ewma.join(std_history, on="symbol", how="inner")
        scaled_trend = combined[ewma_column] / combined[std_column]

        top_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in combined.columns:
                continue
            value = float(scaled_trend[symbol].to_list()[-1])
            if value >= max(scaled_trend.get_symbol().to_list()):
                top_symbols.append(symbol)

        top_symbols = top_symbols[: self._positions]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest