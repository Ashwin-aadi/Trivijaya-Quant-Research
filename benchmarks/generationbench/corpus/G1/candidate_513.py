from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are typically less risky and may offer better risk-adjusted returns. "
        "By tilting our portfolio towards these stocks, we aim to reduce overall portfolio volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or history.width == 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        closes = history[symbols].select(
            pl.col("symbol").alias("symbol"),
            (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean()
        ).collect()

        volatilities: list[float] = [float(v) for v in closes["adj_close"].to_list()]
        sorted_symbols = [s for _, s in sorted(zip(volatilities, symbols))]
        
        weight = 1.0 / self._window
        weights = {symbol: weight for symbol in sorted_symbols[:self._window]}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] if s in weights else 0.0 for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest