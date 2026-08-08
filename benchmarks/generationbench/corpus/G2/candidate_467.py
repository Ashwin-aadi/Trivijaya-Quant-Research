from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeRSIStochastic(Strategy):
    rationale = (
        "The combination of Relative Strength Index (RSI) and Stochastic Oscillator is used to "
        "identify overbought or oversold conditions. When both indicators show extreme values, it "
        "suggests a potential trend reversal."
    )

    def __init__(self, rsi_window: int = 14, stochastic_k_window: int = 5) -> None:
        self._rsi_window = rsi_window
        self._stochastic_k_window = stochastic_k_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._rsi_window, self._stochastic_k_window))
        if closes.height < max(self._rsi_window, self._stochastic_k_window):
            return Signal(information_available_at=stamp, weights={})

        symbols_with_rsi = []
        for symbol in view.symbols:
            rsi_values = _compute_rsi(closes[symbol], window=self._rsi_window)
            k_values = _compute_stochastic_k(closes[symbol], window=self._stochastic_k_window)

            if all(rsi_values[-1] > 0.9) and any(k_values < 20):
                symbols_with_rsi.append(symbol)

        if not symbols_with_rsi:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(symbols_with_rsi)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_with_rsi}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(price_series: pl.Series, window: int) -> list[float]:
    delta = price_series.diff().drop_nulls()
    gain = (delta.where(delta > 0).mean() * 100.0) / price_series.shift(1).mean()
    loss = (-delta.where(delta < 0).mean() * 100.0) / price_series.shift(1).mean()

    rs = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return [float(v) for v in rsi.to_list()[-window:]]


def _compute_stochastic_k(price_series: pl.Series, window: int) -> list[float]:
    high_window = price_series.rolling_max(window=window)
    low_window = price_series.rolling_min(window=window)

    k_values = (price_series - low_window) / (high_window - low_window) * 100
    return [float(v) for v in k_values.to_list()[-window:]]