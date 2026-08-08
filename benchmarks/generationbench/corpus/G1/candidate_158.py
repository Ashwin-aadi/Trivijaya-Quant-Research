from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy seeks to identify and follow trends by measuring volatility. "
        "High volatility periods are indicative of strong market movement, which can be used "
        "to enter positions in the direction of the trend."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        df = history.filter(pl.col("symbol").is_in(symbols))
        returns = (df.select(pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)).with_columns(
            symbol=pl.col("symbol")
        ).pivot(values="adj_close", index="session_date", columns="symbol")

        means = returns.groupby("symbol").agg(pl.col("adj_close").mean().alias("mean"))
        stds = returns.groupby("symbol").agg((pl.col("adj_close") - pl.col("mean")).std().alias("std"))

        combined = means.join(stds, on="symbol")
        combined = combined.with_columns(
            (pl.col("std") / combined.height).alias("volatility_scaled_std")
        )

        breakout_symbols = []
        for symbol in symbols:
            value = float(combined.filter(pl.col("symbol") == symbol)["volatility_scaled_std"])
            if value > self._threshold:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest