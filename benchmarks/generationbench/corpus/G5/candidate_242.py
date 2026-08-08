from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Trends in asset prices are more likely to continue than reverse. By scaling our "
        "position based on recent volatility, we can take larger positions during periods of "
        "high volatility and smaller ones during low volatility."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.5) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = {symbol: float(close) for symbol, close in view.latest_close().items()}
        trend_strengths = {
            symbol: (close / last_close - 1.0) * self._scaling_factor
            for symbol, close in latest_closes.items()
            if (last_close := history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-2]) is not None
        }

        sorted_symbols = [symbol for symbol, _ in sorted(trend_strengths.items(), key=lambda x: abs(x[1]), reverse=True)]
        top_n_symbols = sorted_symbols[:5]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        total_weight = 0.0
        weights = {symbol: (self._scaling_factor / len(top_n_symbols)) for symbol in top_n_symbols}
        for symbol in top_n_symbols:
            weight = abs(weights[symbol])
            if close := latest_closes.get(symbol):
                last_close = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-2]
                weights[symbol] *= (close / last_close - 1.0) * self._scaling_factor
                total_weight += weights[symbol]

        for symbol in top_n_symbols:
            if (weights[symbol] != 0):
                weights[symbol] /= total_weight

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w != 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest