from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the observation that assets with higher "
        "volatility tend to have more pronounced trends. By scaling trades based on recent "
        "volatility, one can capture these trends while potentially reducing risk compared "
        "to a simple momentum strategy."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            close_series = history.select(pl.col("adj_close")[symbol]).to_pandas()[0]
            volatility = _rolling_stddev(close_series, self._volatility_window)
            volatilities[symbol] = volatility

        sorted_symbols = [
            (symbol, volatilities[symbol])
            for symbol in view.symbols
            if symbol in volatilities
        ]
        sorted_symbols.sort(key=lambda x: -x[1])

        weights = {}
        for i, (symbol, _) in enumerate(sorted_symbols):
            weights[symbol] = 1.0 / (i + 1)

        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if v > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rolling_stddev(series: list[float], window: int) -> float:
    mean = sum(series[-window:]) / window
    variance = sum((x - mean) ** 2 for x in series[-window:]) / window
    return variance**0.5 if variance > 0 else 0.0