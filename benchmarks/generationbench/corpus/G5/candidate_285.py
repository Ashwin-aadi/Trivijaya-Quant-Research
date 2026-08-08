from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Equities with lower historical volatility tend to outperform those with higher "
        "volatility over the long term. By tilting towards low-volatility stocks, we aim to "
        "reduce overall portfolio risk while potentially increasing returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[float] = []
        for symbol in symbols:
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_prices) < 2:
                continue
            returns = [(close_prices[i] - close_prices[i-1]) / max(1e-6, close_prices[i-1])
                       for i in range(1, len(close_prices))]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append(volatility)

        ranked_symbols = [s[0] for s in sorted(zip(symbols, volatilities), key=lambda x: x[1])]
        top_n_symbols = ranked_symbols[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest