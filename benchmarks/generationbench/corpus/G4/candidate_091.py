from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrade(Strategy):
    rationale = (
        "This strategy exploits the theme of 'dispersion or range compression' by identifying "
        "periods when stock prices are either widely dispersed (high dispersion) or tightly clustered "
        "(range compression). High dispersion often signals increased uncertainty and potential for "
        "rapid price movements, while range compression suggests stable market conditions with lower "
        "volatility. The rule is to enter trades when the ATR falls below its 20-day moving average, "
        "indicating potential range compression, and exit once ATR rises above this level, signaling "
        "increased dispersion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        atr_values = []
        for symbol in view.symbols:
            high = history.select(pl.col(symbol).alias("high"))
            low = history.select(pl.col(symbol).alias("low"))
            close_lag = history.with_columns(
                (pl.col(symbol).shift(1)).alias(f"{symbol}_close_lag")
            )

            true_range = (
                pl.concat([high, low])
                .max(axis=1)
                .to_series()
                .zip(close_lag[f"{symbol}_close_lag"])
                .map(lambda x: max(abs(x[0] - x[1]), abs(x[0] - x[0].shift(1)), abs(x[1] - x[0].shift(1))))
            )

            atr = true_range.rolling_mean(window_size=self._window, center=True)
            atr_values.append(atr.to_series())

        atr_df = pl.DataFrame({symbol: values for symbol, values in zip(view.symbols, atr_values)})
        atr = atr_df.mean(axis=0).to_numpy()[0]
        atr_mavg = (history.select(pl.col("adj_close").rolling_max(window=self._window)) - 
                    history.select(pl.col("adj_close").rolling_min(window=self._window))) / self._window

        if atr < atr_mavg:
            weight = 1.0 / len(view.symbols)
            return Signal(information_available_at=stamp, weights={s: weight for s in view.symbols})

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest