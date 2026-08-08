from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility indicates a market in flux, potentially creating opportunities for "
        "trend-following strategies. By scaling trades based on historical volatility, the "
        "strategy aims to capitalize on periods of increased price movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

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
            .drop_nulls()
            .sort("session_date")
        )

        # Compute volatility as standard deviation of returns
        vol = history["r"].std()

        # Scale weights based on volatility
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = [float(v) for v in history[symbol].to_list()]
            if len(daily_returns) < self._window:
                continue
            scaled_weight = vol * 0.1  # Arbitrarily choosing a scaling factor of 0.1
            picks.append(symbol)

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