from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have less unexpected price movements and, on average, "
        "deliver higher returns over time. This strategy tilts the portfolio towards low-volatility"
        " stocks in an attempt to capture these excess returns."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["symbol"])
        )

        # Calculate historical volatility
        volatilities = (
            history_with_returns.group_by("symbol")
            .agg(
                (pl.col("return").std().alias("volatility"))
            )
            .sort("volatility")
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Select top N low-volatility symbols
        low_vol_symbols = [str(sym[0]) for sym in volatilities.select("symbol").head(10)]
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