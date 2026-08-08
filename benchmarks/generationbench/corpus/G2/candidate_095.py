from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy exploits a composite signal based on both the 50-day moving average "
        "and the relative strength index (RSI). The idea is that stocks often show stronger "
        "performance when they are above their 50-day moving average and in an overbought or "
        "oversold state as measured by RSI."
    )

    def __init__(self, ma_window: int = 50, rsi_window: int = 14) -> None:
        self._ma_window = ma_window
        self._rsi_window = rsi_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._ma_window, self._rsi_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        closes = view.closes(lookback=self._ma_window)
        rsi_history = _compute_rsi(history, self._rsi_window)

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in symbols:
            ma_value = closes[symbol].to_list()[-1]
            rsi_value = rsi_history[symbol].to_list()[-1]

            if history["adj_close"][history.height - 1] > ma_value and rsi_value >= 70 or \
               history["adj_close"][history.height - 1] < ma_value and rsi_value <= 30:
                signals[symbol] = 1.0 / len(symbols)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_rsi(history: pl.DataFrame, window: int) -> pl.DataFrame:
    closes = history.select(pl.col("adj_close")).to_pandas()
    rsi_values = []

    for symbol in closes.columns[1:]:
        delta = closes[f"{symbol}"].diff().dropna()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling_mean(window).to_list()[window - 1:]
        avg_loss = loss.rolling_mean(window).to_list()[window - 1:]

        rs = [gain_i / loss_i if loss_i != 0 else 0 for gain_i, loss_i in zip(avg_gain, avg_loss)]
        rsi = [100 - (100 / (1 + x)) for x in rs]

        rsi_values.append(rsi)

    return pl.DataFrame({symbol: values for symbol, values in zip(closes.columns[1:], rsi_values)})