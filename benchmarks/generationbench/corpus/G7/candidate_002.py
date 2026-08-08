from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends based on exponential moving averages (EMAs) of closing "
        "prices, scaled by the historical volatility. The combination aims to capture both trend "
        "momentum and market volatility, providing a balanced approach to equity trading."
    )

    def __init__(self, ema_window: int = 20, vol_window: int = 20) -> None:
        self._ema_window = ema_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._ema_window + self._vol_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        ema_column = f"ema_{self._ema_window}"
        vol_column = f"vol_{self._vol_window}"

        history = (
            history
            .with_columns(
                (pl.col("close").rolling_mean(window_size=self._ema_window)).alias(ema_column),
                ((pl.col("adj_close") - pl.col("adj_close").shift(1)) ** 2)
                .rolling_sum(window_size=self._vol_window)
                .mean()
                .sqrt()
                .alias(vol_column),
            )
        )

        signals = []
        for symbol in symbols:
            if history.filter(pl.col("symbol") == symbol).height < self._ema_window + self._vol_window - 1:
                continue
            ema_value = float(history.filter(pl.col("symbol") == symbol)[ema_column][-1])
            vol_value = float(history.filter(pl.col("symbol") == symbol)[vol_column][-1])

            if ema_value > history.filter(pl.col("symbol") == symbol)["close"][-2]:
                signals.append(symbol)

        weight = 1.0 / len(symbols) if signals else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols if s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest