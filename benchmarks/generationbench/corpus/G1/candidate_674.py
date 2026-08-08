from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting position sizes "
        "based on recent volatility. High volatility periods suggest increased risk and thus smaller "
        "positions should be taken, whereas low volatility suggests a stronger trend and larger positions."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window)
        if history.height < self._window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = view.closes()
        symbols = [symbol for symbol in view.symbols if symbol in recent_closes.columns]

        volatilities: dict[str, float] = {}
        for symbol in symbols:
            prices = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            log_returns = [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]
            volatility = (sum([r**2 for r in log_returns])**0.5) / ((len(log_returns) - 1) * self._vol_window)**0.5
            volatilities[symbol] = float(volatility)

        trend_scores: dict[str, float] = {}
        for symbol in symbols:
            latest_close = view.latest_close()[symbol]
            close_series = history.filter(pl.col("symbol") == symbol).select(["session_date", "adj_close"])
            recent_closes_list = [float(c) for c in close_series["adj_close"].to_list()]
            max_close = max(recent_closes_list)
            if abs(latest_close - max_close) / max_close <= 0.1:  # 10% deviation
                trend_scores[symbol] = volatilities[symbol]

        picks: list[str] = [symbol for symbol, score in trend_scores.items() if score > 0]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest