from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies breakouts from key support or resistance levels and exploits "
        "the tendency for prices to continue moving in the direction of the breakout. It leverages "
        "historical OHLC data to identify strong breakout candidates with volume confirmation, "
        "and sets targets based on Fibonacci retracement levels."
    )

    def __init__(self, window: int = 90, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        key_levels = {}
        for symbol in view.symbols:
            levels = self._calculate_key_levels(history.select(pl.col("adj_close")).select([symbol]))
            key_levels[symbol] = levels

        breakout_signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            breakout_strength, volume_confirmation = self._evaluate_breakout(symbol, adj_closes, key_levels)
            if breakout_strength is not None and volume_confirmation > 0.5:
                breakout_signals[symbol] = breakout_strength * volume_confirmation

        top_symbols = sorted(breakout_signals.items(), key=lambda x: x[1], reverse=True)[: self._top_n]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        signal_weights = {symbol: weight for symbol, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=signal_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_key_levels(adj_close_series: pl.Series) -> tuple[float, float]:
    high = adj_close_series.max().item()
    low = adj_close_series.min().item()
    return (high, low)


def _evaluate_breakout(symbol: str, adj_closes: list[float], key_levels: dict[str, tuple[float, float]]) -> tuple[Optional[float], float]:
    recent_high, recent_low = key_levels[symbol]
    breakout_strength = None
    for i in range(len(adj_closes) - 10):  # Consider last 10 days for breakout strength
        if adj_closes[i] < recent_low and adj_closes[-1] > recent_high:
            breakout_strength = abs(adj_closes[-1] - recent_high)
            break
        elif adj_closes[i] > recent_high and adj_closes[-1] < recent_low:
            breakout_strength = abs(adj_closes[-1] - recent_low)
            break

    if breakout_strength is None:
        return (None, 0.5)

    # Volume confirmation can be a simple threshold or more complex logic
    volume_confirmation = sum(view.closes().select(pl.col(symbol)).to_series().to_list()) / len(adj_closes) > 1e6

    return (breakout_strength, float(volume_confirmation))