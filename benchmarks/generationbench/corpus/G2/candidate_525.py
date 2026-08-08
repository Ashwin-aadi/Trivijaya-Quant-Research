from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This is due to the risk-reward tradeoff where lower volatility often implies a higher "
        "expected return."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            (history["adj_close"] / history["adj_close"].shift(1) - 1.0)
            .select(pl.all().mean())
            .to_dict()[0]
        )
        volatilities = [pl.col(f"adj_close").std().item() for f in symbols]

        sorted_indices = sorted(range(len(symbols)), key=lambda i: volatilities[i])
        top_symbols = [symbols[i] for i in sorted_indices[:5]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest