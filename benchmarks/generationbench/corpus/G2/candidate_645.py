from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility ones over the long term "
        "due to reduced risk and potentially higher dividend yields. By tilting our portfolio "
        "towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate historical returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        # Group by symbol and calculate mean return as a proxy for volatility
        volatilities = (
            history.group_by("symbol")
            .agg(
                (pl.col("return").abs().mean().alias("volatility"))
            )
            .select("symbol", "volatility")
        )

        # Get the latest close prices to determine the most recent data available
        closes = view.closes(lookback=self._window)
        latest_closes = {symbol: float(close) for symbol, close in closes.to_dict().items()}

        # Select symbols with lowest volatilities
        low_vol_symbols = [
            symbol for symbol, volatility in volatilities.sort("volatility").to_dict(False).items()
            if symbol in latest_closes and latest_closes[symbol] is not None
        ][:5]

        # Assign equal weights to the selected symbols
        weight = 1.0 / len(low_vol_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in low_vol_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest