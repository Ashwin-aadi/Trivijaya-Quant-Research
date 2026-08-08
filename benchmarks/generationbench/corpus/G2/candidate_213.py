from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower idiosyncratic risk and may outperform the broader market "
        "over long periods due to their more stable price movements. This strategy aims to capture these "
        "stability benefits by overweighting low-volatility stocks in the portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            daily_returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = (sum([r**2 for r in daily_returns]) / len(daily_returns)) ** 0.5
            symbol_volatility[symbol] = volatility

        sorted_symbols = sorted(symbol_volatility.items(), key=lambda x: x[1])
        top_low_volatility = [s for s, v in sorted_symbols[:3]]  # Select the lowest 3 volatilities
        weight = 1.0 / len(top_low_volatility)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_low_volatility}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest