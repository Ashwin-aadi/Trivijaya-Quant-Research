from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low-volatility stocks, we aim to capture these "
        "enhanced returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbol_list) < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in symbol_list:
            prices = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_price")
            ).to_pandas()
            returns = (
                (prices[f"{symbol}_price"].pct_change().dropna() * 100)
                .mean()
                .values[0]
            )
            volatility = prices[f"{symbol}_price"].std().values[0] * 100
            volatilities[symbol] = volatility

        sorted_symbols = [
            symbol for _, symbol in sorted(volatilities.items(), key=lambda item: item[1])
        ][:5]

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest