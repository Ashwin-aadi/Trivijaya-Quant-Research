from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks are often less risky and can offer more stable returns over "
        "time. By tilting our portfolio towards these stocks, we may capture the low-volatility"
        " anomaly where lower risk is associated with higher returns."
    )

    def __init__(self, lookback_window: int = 60) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols]
        history["return"] = (
            (history["adj_close"] / history["adj_close"].shift(1)) - 1.0
        ).drop_nulls().cast(pl.Float64)

        # Calculate the standard deviation of returns over the lookback period
        std_devs = (
            history.groupby("symbol")
                   .agg(pl.col("return").std().alias("std_dev"))
                   .sort("std_dev", descending=False)
        )

        if std_devs.height < len(symbol_list):
            return Signal(information_available_at=stamp, weights={})

        # Select the lowest volatility symbols
        low_vol_symbols = [symbol for symbol in std_devs["symbol"].to_list()[:5]]

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