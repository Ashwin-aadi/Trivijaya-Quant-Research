from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are less likely to experience sharp price movements and "
        "may offer more stable returns. By tilting towards these stocks, the portfolio can "
        "reduce overall risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] if i > 0 else 0.0
                       for i in range(len(prices))]
            volatility = (sum(abs(r) for r in returns) / len(returns))
            volatility_scores.append((symbol, volatility))

        volatility_scores.sort(key=lambda x: x[1])
        picks = [s[0] for s in volatility_scores[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest