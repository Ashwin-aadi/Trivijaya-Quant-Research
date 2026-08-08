from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrend(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to capture trends while adjusting the "
        "position size based on recent volatility. Higher volatility suggests that prices are "
        "more likely to continue trending, so we should increase our position size."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).drop_nulls()
        mean_return = returns.mean().item()
        std_dev = returns.std().item()

        scaled_weights: dict[str, float] = {}
        for symbol in symbols:
            recent_returns = returns[symbol].to_list()[-self._window:]
            mean_r, std_r = float(mean_return[symbol]), float(std_dev[symbol])
            if std_r == 0:
                continue
            weight = self._scale_factor * (mean_r / std_r)
            scaled_weights[symbol] = min(max(weight, -1.0), 1.0)

        non_zero_symbols = [s for s in symbols if abs(scaled_weights.get(s, 0)) > 0]
        if not non_zero_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        total_weight = sum(scaled_weights.values())
        adjusted_weights = {s: w / total_weight for s, w in scaled_weights.items()}
        return Signal(
            information_available_at=stamp,
            weights={s: adjusted_weights[s] for s in non_zero_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest