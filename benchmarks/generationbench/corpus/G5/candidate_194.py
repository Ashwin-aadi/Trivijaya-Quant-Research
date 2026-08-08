from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the relationship between a stock's "
        "recent volatility and its price movement. Higher volatility indicates increased "
        "market uncertainty or noise, which can lead to mean reversion. By scaling trades "
        "by recent volatility, we aim to capture trends more effectively."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .with_column(pl.col("r").cumsum().over("symbol").alias("ret"))
        )

        # Calculate volatility
        vol = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().over("session_date") * self._scale_factor).alias(
                    "vol"
                )
            )
            .with_columns(pl.lit(stamp.date()).alias("information_available_at"))
        )

        # Generate signals
        picks = {}
        for symbol in view.symbols:
            if symbol not in vol.columns or vol.height < 1:
                continue

            latest_volatility = float(vol[vol["symbol"] == symbol]["vol"].to_list()[0])
            ret = history.filter(pl.col("symbol") == symbol).select("ret").tail(1).item()
            if ret > latest_volatility / self._scale_factor * 2.0:
                picks[symbol] = 1.0 / len(view.symbols)

        return Signal(information_available_at=stamp, weights=picks)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest