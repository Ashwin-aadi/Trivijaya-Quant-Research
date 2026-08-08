from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over long periods. "
        "By tilting our portfolio towards these stocks, we can potentially reduce risk and enhance returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(symbol) for symbol in view.symbols]
        volatilities: list[float] = []
        for symbol in symbols:
            df = history.select(
                pl.col("session_date"),
                (pl.col("adj_close").shift_and_fill(1) / pl.col("adj_close") - 1).alias("returns")
            ).sort("session_date")
            returns = df["returns"].to_list()[1:]
            vol = (sum([r**2 for r in returns])**0.5) / len(returns)
            volatilities.append(float(vol))

        ranked_symbols = [symbols[i] for i in sorted(range(len(symbols)), key=lambda x: volatilities[x])[:5]]
        weights = {s: 1.0/len(ranked_symbols) for s in ranked_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest