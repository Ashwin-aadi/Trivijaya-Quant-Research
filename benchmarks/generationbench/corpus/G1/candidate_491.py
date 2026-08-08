from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeAndVolatility(Strategy):
    rationale = (
        "This strategy selects stocks that show high volume and low volatility, "
        "indicating potential liquidity and reduced price fluctuation risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = _filter_high_volume(history)
        low_volatility_symbols = _filter_low_volatility(history)

        picks: list[str] = set(high_volume_symbols).intersection(low_volatility_symbols)
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _filter_high_volume(history: pl.DataFrame) -> list[str]:
    high_volume_symbols = []
    for symbol in view.symbols:
        if symbol not in history.columns:
            continue
        volume_series = [float(v) for v in history[symbol].col("volume").to_list()]
        if len(volume_series) < 20:
            continue
        max_volume = max(volume_series)
        if any(v == max_volume for v in volume_series[-5:]):
            high_volume_symbols.append(symbol)
    return high_volume_symbols


def _filter_low_volatility(history: pl.DataFrame) -> list[str]:
    low_volatility_symbols = []
    for symbol in view.symbols:
        if symbol not in history.columns:
            continue
        close_series = [float(v) for v in history[symbol].col("adj_close").to_list()]
        if len(close_series) < 20:
            continue
        daily_returns = [(close / close.shift(1) - 1.0) for close in close_series[1:]]
        std_dev = pl.Series(daily_returns).std()
        if std_dev <= 0.05:
            low_volatility_symbols.append(symbol)
    return low_volatility_symbols


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest