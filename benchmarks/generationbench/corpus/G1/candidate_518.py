from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy uses volatility scaling to determine trend following signals. "
        "High volatility periods are treated with caution while low volatility periods "
        "are exploited for trend-following trades."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window - 1)

        if history.height < self._window + self._vol_window - 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(pl.col("adj_close").tail(self._window))
        vol_history = history.select(pl.col("adj_close").tail(self._vol_window))

        if recent_closes.height < self._window or vol_history.height < self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = recent_closes.mean().item()
        vol = (vol_history - vol_history.shift(1)).abs().mean().item()

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            trend_score = (history[symbol].tail(self._window) - mean_close) / vol
            max_trend_score = max(trend_score.to_list())
            min_trend_score = min(trend_score.to_list())

            if max_trend_score > 0.5 or min_trend_score < -0.5:
                trends[symbol] = trend_score[-1]

        if not trends:
            return Signal(information_available_at=stamp, weights={})

        sorted_trends = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)
        weight = 1.0 / len(sorted_trends)

        signal_weights = {symbol: weight for symbol, _ in sorted_trends}
        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest