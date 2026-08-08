from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term due "
        "to risk-averse investor behavior and market inefficiencies. This strategy exploits this "
        "phenomenon by constructing a portfolio weighted towards low-volatility assets, aiming to "
        "capture the associated premium."
    )

    def __init__(self, window: int = 60, num_stocks: int = 50) -> None:
        self._window = window
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate log returns
        returns = (
            history.select(
                [
                    "symbol",
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return"),
                ]
            )
            .drop_nulls()
            .filter(pl.col("log_return") != 0)
        )

        # Calculate historical volatility
        volatilities = (
            returns.groupby("symbol")
            .agg(
                (pl.col("log_return").std().alias("volatility")),
            )
            .sort("volatility", descending=False)
        ).to_pandas()

        if len(volatilities) < self._num_stocks:
            return Signal(information_available_at=stamp, weights={})

        # Rank stocks based on volatility
        rank = volatilities.reset_index().set_index("symbol").squeeze().rank(method="min", ascending=True)

        # Select top num_stocks based on inverse rank
        selected_stocks = rank.nlargest(self._num_stocks).index.to_list()

        weights = {s: 1.0 / len(selected_stocks) for s in selected_stocks}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest