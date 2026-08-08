from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time due to "
        "reduced uncertainty and potentially higher risk-adjusted returns. By tilting the portfolio towards lower volatility stocks, we can benefit from this systematic anomaly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns and s in closes.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        ).collect()

        # Calculate the historical volatility for each stock
        volatilities = (
            history.lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").std().alias("volatility"))
        ).collect()

        # Sort symbols by volatility
        sorted_symbols = volatilities.sort("volatility", descending=False)["symbol"].to_list()
        
        weights: dict[str, float] = {}
        weight_per_symbol = 1.0 / len(sorted_symbols)
        for symbol in sorted_symbols:
            weights[symbol] = weight_per_symbol

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest