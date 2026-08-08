from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedBreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies breakouts from support or resistance levels to capture continuation trends. "
        "It uses daily high/low prices combined with Bollinger Bands for breakout confirmation, ensuring reliability through both technical and statistical measures."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        bollinger_bands = _calculate_bollinger_bands(history)
        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in bollinger_bands.columns:
                continue
            values = [float(v) for v in bollinger_bands[symbol].to_list()]
            if len(values) < self._window:
                continue

            last_close = history.height - 1
            latest_close = float(history["adj_close"][last_close][symbol])
            lower_band, upper_band = values[-2], values[-1]

            if (latest_close > upper_band and bollinger_bands[symbol].stddev() < 0.5) or \
                    (latest_close < lower_band and bollinger_bands[symbol].stddev() < 0.5):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_bollinger_bands(history: pl.DataFrame) -> pl.DataFrame:
    history = history.with_columns(
        (pl.col("adj_close") - pl.col("adj_close").rolling_mean(window_size=20)).alias("deviation")
    )
    std_dev_column = (history["deviation"] / pl.col("adj_close").rolling_std(window_size=20)).alias("stddev")
    history = history.with_columns(std_dev_column)
    return history