from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower downside risk and often provide more stable "
        "returns over the long term. Tilting towards low volatility can help reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_series = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").std().over(pl.arange(1, pl.count() + 1)).last())
                .alias("volatility")
            )
            .collect()
        )

        symbols = [s for s in view.symbols if s in volatility_series.columns]
        sorted_symbols = (
            volatility_series.sort("volatility", descending=False)["symbol"].to_list()
        )[: len(symbols)]

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