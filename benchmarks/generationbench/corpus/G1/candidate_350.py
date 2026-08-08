from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling trades based on "
        "recent volatility. High volatility periods suggest potential for significant price movement, "
        "prompting larger positions during such times."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.0) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        # Calculate rolling standard deviation of returns as a proxy for volatility
        vol = (
            history.select(pl.col("return").rolling_std(window=self._window))
            .rename({"return": "volatility"})
            .with_columns((pl.col("volatility") * self._scale_factor).alias("scaled_vol"))
        )

        # Determine top symbols based on scaled volatility
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in vol.columns or float(vol[symbol].max()) == 0.0:
                continue
            picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Equal weighting among selected symbols
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