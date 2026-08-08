from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often perceived to be less risky and can offer more stable returns. "
        "By tilting our portfolio towards low-volatility stocks, we aim to reduce overall risk while maintaining competitiveness in returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        )
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the volatility for each stock
        volatilities: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = history.filter(pl.col("symbol") == symbol)["returns"].to_list()
            if not returns:
                continue
            volatility = (sum(returns**2) / self._window) ** 0.5
            volatilities.append(volatility)

        # Get the symbols with the lowest volatilities
        sorted_symbols = [
            s for _, s in sorted(zip(volatilities, view.symbols), key=lambda x: x[0])
        ][:5]
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest