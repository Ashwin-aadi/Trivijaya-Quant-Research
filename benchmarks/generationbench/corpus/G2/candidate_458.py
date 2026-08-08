from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the idea that assets with higher recent "
        "volatility are more likely to continue trending in their direction of movement. By "
        "scaling our position based on volatility, we can capture more significant trends while "
        "remaining cautious during periods of low volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["symbol"])
        )

        # Compute volatility
        volatilities = history.groupby("symbol").agg(
            (pl.col("return").std().alias("volatility"))
        ).collect()

        # Scale weights by volatility
        scaled_weights = {}
        total_volatility = 0.0
        for symbol in view.symbols:
            if symbol not in volatilities.column_names:
                continue
            volatility = float(volatilities.filter(pl.col("symbol") == symbol)["volatility"])
            total_volatility += volatility
            scaled_weights[symbol] = volatility

        # Normalize weights to sum up to 1.0 (or close to it)
        if total_volatility > 0:
            for symbol in view.symbols:
                if symbol not in scaled_weights:
                    continue
                scaled_weights[symbol] /= total_volatility

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in scaled_weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest