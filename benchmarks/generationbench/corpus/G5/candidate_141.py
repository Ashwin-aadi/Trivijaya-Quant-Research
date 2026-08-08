from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date")
            .select(["symbol", "session_date", "r"])
        )

        # Calculate historical volatility
        volatilities = (
            history.group_by("symbol")
            .agg(pl.col("r").std().alias("volatility"))
            .sort(by="volatility")
        )

        if volatilities.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Select top N low-volatility stocks
        num_low_vol_symbols = min(len(view.symbols), 5)
        low_vol_symbols = [row["symbol"] for row in volatilities.to_dicts()[:num_low_vol_symbols]]

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest