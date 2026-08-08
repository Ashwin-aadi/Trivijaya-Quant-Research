from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeRSIEMA(Strategy):
    rationale = (
        "Combining Relative Strength Index (RSI) with Exponential Moving Average (EMA) can "
        "provide a more nuanced view of stock strength and momentum. RSI indicates overbought or "
        "oversold conditions, while EMA captures recent price action trends."
    )

    def __init__(self, rsi_window: int = 14, ema_period: int = 50) -> None:
        self._rsi_window = rsi_window
        self._ema_period = ema_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._rsi_window, self._ema_period))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(closes) < self._rsi_window + self._ema_period - 1:
                continue

            # Calculate RSI
            deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            gains = [d if d > 0 else 0.0 for d in deltas]
            losses = [-l if l > 0 else 0.0 for l in deltas]

            avg_gain = sum(gains[:self._rsi_window]) / self._rsi_window
            avg_loss = sum(losses[:self._rsi_window]) / self._rsi_window

            gains_exp = [gains[i] + (avg_gain * 1) if i < self._rsi_window else gains[i] for i in range(len(gains))]
            losses_exp = [losses[i] + (avg_loss * 1) if i < self._rsi_window else losses[i] for i in range(len(losses))]

            rs = sum(gains_exp[-self._rsi_window:]) / sum(losses_exp[-self._rsi_window:]) if avg_loss > 0 else 0.0
            rsi = 100 - (100 / (1 + rs))

            # Calculate EMA
            ema_close = _ema(closes, self._ema_period)

            # Composite score: higher RSI and lower EMA favor buying
            if rsi > 70 and ema_close < closes[-1]:
                signals[symbol] = 1.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def _ema(data: list[float], period: int) -> float:
    n = len(data)
    ema = [data[0]] * (period - 1) + data
    for i in range(period, n):
        ema.append(2 * (data[i] - ema[i-1]) / period + ema[i-1])
    return sum(ema[-period:]) / period