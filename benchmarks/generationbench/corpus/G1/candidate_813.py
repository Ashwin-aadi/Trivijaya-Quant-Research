from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered to be less risky and can provide more stable returns. "
        "By tilting the portfolio towards low-volatility stocks, we aim to reduce overall risk while still aiming for positive returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if len(symbol_list) < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history
            .filter(pl.col("session_date") > (view.as_of - pl.duration(days=self._window)))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility"))
            )
            .sort("volatility", descending=False)
        )

        top_symbols = volatilities["symbol"].to_list()[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest