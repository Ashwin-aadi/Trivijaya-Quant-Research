from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and have historically provided "
        "positive abnormal returns. This strategy aims to tilt the portfolio towards low-volatility "
        "stocks to capture this premium."
    )

    def __init__(self, lookback_window: int = 60) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        volatilities = {symbol: 0.0 for symbol in symbols}

        for symbol in symbols:
            if symbol not in history.columns:
                continue

            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._lookback_window:
                continue

            returns = [(prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8) for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in returns])**0.5) / self._lookback_window
            volatilities[symbol] = volatility

        sorted_symbols = [symbol for symbol in symbols if symbol in volatilities]
        sorted_symbols.sort(key=lambda s: volatilities[s])
        
        top_n_low_vol = 5  # Number of low-volatility stocks to select
        selected_symbols = sorted_symbols[:top_n_low_vol]

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest