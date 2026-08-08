from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "This strategy is based on volatility-scaled trend following. It aims to capture "
        "trends by using a moving average of returns and adjusting weights according to the "
        "volatility of each asset."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility = {}
        for symbol in view.symbols:
            daily_returns = (
                (history[pl.col("symbol") == symbol]["adj_close"].to_list()[1:]
                 / history[pl.col("symbol") == symbol]["adj_close"].shift(1).to_list()[:-1])
                - 1.0
            )
            volatility = pl.Series(daily_returns).std()
            if not pl.is_nan(volatility):
                symbol_volatility[symbol] = volatility

        top_symbols = sorted(symbol_volatility, key=lambda x: abs(symbol_volatility[x]), reverse=True)[:5]
        weights = {}
        for symbol in top_symbols:
            windowed_returns = history[pl.col("symbol") == symbol]["adj_close"].to_list()[1:]
            trend = pl.Series(windowed_returns).mean()
            if abs(trend) >= self._threshold * symbol_volatility[symbol]:
                weight = 1.0 / len(top_symbols)
                weights[symbol] = max(min(weight, 1.0), -1.0)

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest