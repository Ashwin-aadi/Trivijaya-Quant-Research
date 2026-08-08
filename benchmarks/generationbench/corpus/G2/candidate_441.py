from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and can have better risk-adjusted "
        "returns. By tilting our portfolio towards low-volatility stocks, we aim to capture the "
        "outperformance typically associated with this style."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate daily returns
        rets = (
            history
            .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("ret"))
            .sort("session_date")
            .select(["symbol", "session_date", "ret"])
        )

        # Calculate historical volatility for each stock
        volatilities = (
            rets.groupby("symbol")
            .agg(
                (pl.col("ret").std().alias("volatility")),
            )
            .collect()
        ).to_dict(False)

        # Filter out symbols without enough data
        valid_symbols = [
            symbol
            for symbol in closes.columns
            if symbol in volatilities and len(volatilities[symbol]) >= self._window
        ]

        # Sort by volatility to get the lowest ones
        low_volatility_symbols = sorted(valid_symbols, key=lambda x: volatilities[x])

        # Select top 5 symbols for tilting
        picks = low_volatility_symbols[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest