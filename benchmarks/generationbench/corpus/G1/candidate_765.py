from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are generally perceived to be less risky and may offer more stable returns. "
        "By tilting our portfolio towards these stocks, we aim to capture the risk premium associated with lower volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [s for s in view.symbols if s in history["symbol"].unique().to_list()]
        if len(symbol_list) < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history.filter(pl.col("symbol").is_in(symbol_list))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean().alias("volatility")
            )
            .sort("volatility", descending=False)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volatilities)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volatilities},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest