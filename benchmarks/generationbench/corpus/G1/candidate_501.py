from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low volatility, we aim to capture this risk premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            values = [float(v) for v in history.select(pl.col(symbol)).drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            log_returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, len(values))]
            volatility = (sum([r**2 for r in log_returns]) / len(log_returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = [s for s, v in sorted(volatilities.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().to_list()[0]
    assert isinstance(newest, date)
    return newest