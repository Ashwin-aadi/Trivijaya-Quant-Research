from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time. This is "
        "known as the low volatility anomaly and can be exploited by tilting portfolios towards "
        "low-volatility assets."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [s for s, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_low_volatility = sorted_symbols[:int(len(sorted_symbols) * 0.2)]
        weight = 1.0 / len(top_low_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_low_volatility}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest