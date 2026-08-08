from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over long periods. "
        "By tilting our portfolio towards lower volatility assets, we aim to capture "
        "this effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                (pl.col("volume") * pl.col("adj_close")).sum()
                / pl.col("volume").sum()
                .alias("weighted_avg_price"),
            )
        )

        volatilities = (
            symbol_prices.with_columns(
                (pl.col("adj_close").std() / pl.col("avg_price") * 100).alias("volatility")
            )
            .sort("volatility", descending=False)
            .limit(20)[["symbol"]]
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volatilities)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volatilities["symbol"].to_list()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest