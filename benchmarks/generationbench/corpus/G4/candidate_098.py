from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy leverages trends while scaling them by volatility to adapt to market conditions. "
        "High volatility often precedes significant price movements, allowing for timely entry into established trends."
    )

    def __init__(self, trend_window: int = 50, atr_window: int = 14, max_positions: int = 30) -> None:
        self._trend_window = trend_window
        self._atr_window = atr_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._trend_window + 10, self._atr_window))
        if history.is_empty() or history.height < max(self._trend_window + 10, self._atr_window):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._trend_window)
        atr_values = _compute_atr(history)
        if not all(atr_values.values()):
            return Signal(information_available_at=stamp, weights={})

        # Calculate trend strength for each symbol
        trends = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in atr_values.keys():
                continue
            current_close = float(view.latest_close()[symbol])
            sma_50 = history.select(
                pl.col("symbol").filter(pl.col("symbol") == symbol)
                .select(pl.col("adj_close").mean())
                .to_series()
                .item()
            )
            trend_strength = abs(current_close - sma_50) / atr_values[symbol]
            trends[symbol] = trend_strength

        # Rank symbols by their trend strength
        ranked_symbols = sorted(trends, key=trends.get, reverse=True)[: self._max_positions]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_position = 1.0 / len(ranked_symbols)
        adjusted_weight = _adjust_weight_by_volatility(atr_values)

        return Signal(
            information_available_at=stamp,
            weights={
                s: adjusted_weight * weight_per_position for s in ranked_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_atr(history: pl.DataFrame) -> dict[str, float]:
    high = history.select(pl.col("high"))
    low = history.select(pl.col("low"))
    close = history.select(pl.col("close"))

    true_range = (high - low).to_list() + [0] * len(low)
    prev_close = [float(close.to_list()[0])] + list(close.to_list()[:-1])
    atr = [(max(x, max(y, abs(z - w))) for x, y, z, w in zip(true_range, true_range[1:], low.to_list(), prev_close))]

    return {symbol: float(sum(atr) / len(atr)) for symbol in history["symbol"].to_list()}


def _adjust_weight_by_volatility(atr_values: dict[str, float]) -> float:
    max_atr = max(atr_values.values())
    min_atr = min(atr_values.values())

    if max_atr == min_atr:
        return 1.0

    def adjust_factor(atr_value):
        return (max_atr - atr_value) / (max_atr - min_atr)

    volatility_scaled_weights = {symbol: adjust_factor(atr_value) for symbol, atr_value in atr_values.items()}

    max_weight = max(volatility_scaled_weights.values())
    if max_weight > 1.0:
        return max_weight
    else:
        return 1.0