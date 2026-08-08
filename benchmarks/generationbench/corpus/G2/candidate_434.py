from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFT(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets with higher volatility "
        "are more likely to continue trending in their recent direction. By scaling trends by "
        "volatility, we can capture these momentum effects while mitigating risk."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._volatility_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(
            pl.col("session_date").alias("date"), *[pl.col(symbol).alias(f"close_{symbol}") for symbol in symbols]
        )

        # Calculate simple returns
        returns = closes.with_columns(
            [(pl.col(f"close_{symbol}") / pl.col(f"close_{symbol}").shift(1) - 1.0).alias(f"return_{symbol}") for symbol in symbols]
        )
        returns = returns.sort("date", descending=True)

        # Calculate volatility
        volatilities = returns.select(
            [pl.col(f"return_{symbol}").std().over(pl.col("date").shift(-(self._volatility_window - 1))).alias(f"vol_{symbol}") for symbol in symbols]
        )

        # Scale returns by volatility and pick the top performers
        scaled_returns = volatilities.with_columns(
            [(pl.col(f"return_{symbol}") / pl.col(f"vol_{symbol}")).alias(f"scaled_return_{symbol}") for symbol in symbols]
        )
        top_performers = sorted(
            [symbol for symbol in symbols if f"vol_{symbol}" in scaled_returns.columns and f"scaled_return_{symbol}" in scaled_returns.columns],
            key=lambda s: float(scaled_returns.filter(pl.col(f"scaled_return_{s}").is_not_null()).select(pl.col(f"scaled_return_{s}")).row(0)[0]),
            reverse=True
        )[:5]

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest