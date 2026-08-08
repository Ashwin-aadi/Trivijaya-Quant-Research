from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining a momentum signal with a volatility filter aims to identify stocks that are "
        "both trending upwards and showing reduced price variability. This dual characteristic "
        "could indicate strong underlying fundamentals and lower risk of sharp corrections."
    )

    def __init__(self, momentum_window: int = 20, vol_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.is_empty() or history.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            momentum_signal = _calculate_momentum(symbol, history)
            volatility_signal = _calculate_volatility(symbol, history)

            if momentum_signal and not volatility_signal:
                picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_momentum(symbol: str, history: pl.DataFrame) -> bool:
    closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
    if len(closes) < 2 * _momentum_window:
        return False
    momentum = (closes[-_momentum_window] - closes[0]) / closes[0]
    return momentum > 0.1


def _calculate_volatility(symbol: str, history: pl.DataFrame) -> bool:
    closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
    if len(closes) < 2 * _volatility_window:
        return False
    daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    volatility = (sum(daily_returns**2) / len(daily_returns)) ** 0.5
    return volatility < 0.05