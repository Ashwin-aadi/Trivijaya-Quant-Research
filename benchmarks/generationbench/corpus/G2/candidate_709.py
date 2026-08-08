from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their mean "
        "tend to revert back towards it. In a short horizon, recent high volatility can lead to "
        "prices returning to the mean, offering trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = view.closes(lookback=self._window)

        symbols_with_mean_reversion_signal: list[str] = []
        for symbol in view.symbols:
            if symbol not in recent_closes.columns:
                continue
            values = [float(v) for v in history[["session_date", "symbol", symbol]].filter(
                (pl.col("close") - pl.col(symbol).mean().over("symbol")) < 0.0
            )[symbol].to_list()]
            if len(values) < self._window:
                continue

            mean_deviation = sum(abs(value - history.select(pl.col(symbol)).mean()) for value in values)
            if mean_deviation > 0.5 * abs(history.select(pl.col(symbol)).mean() - recent_closes[symbol].tail(1).item()):
                symbols_with_mean_reversion_signal.append(symbol)

        weights: dict[str, float] = {symbol: 1.0 / len(symbols_with_mean_reversion_signal) for symbol in
                                     symbols_with_mean_reversion_signal}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest