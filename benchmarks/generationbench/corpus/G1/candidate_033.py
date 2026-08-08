from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time. "
        "By tilting our portfolio towards low volatility, we aim to capture this anomaly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        symbol_prices = {sym: float(prices[-1]) for sym, prices in
                         history[symbols].select(pl.col("symbol"), pl.col("adj_close").tail(1)).iter_rows()}
        
        volatility_series = {}
        for symbol in symbols:
            prices = [float(p) for p in view.closes(lookback=self._window)[symbol]]
            if len(prices) < self._window:
                continue
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility_series[symbol] = sum([r**2 for r in returns])**0.5

        sorted_symbols = [symbol for symbol, _ in sorted(volatility_series.items(), key=lambda item: item[1])]
        
        top_5_low_volatility = sorted_symbols[:5]
        weight = 1.0 / len(top_5_low_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_5_low_volatility}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest