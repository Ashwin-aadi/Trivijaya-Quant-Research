from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower downside risk and can provide more stable returns. "
        "By tilting our portfolio towards these stocks, we aim to reduce overall portfolio volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        mean_volatility = (
            history.lazy()
            .group_by("symbol")
            .agg((pl.col("adj_close").std().alias("volatility")))
            .sort("volatility", descending=False)
            .select(["symbol"])
            .to_pandas()["symbol"]
            .tolist()[: len(symbols)]
        )

        if not mean_volatility:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(mean_volatility)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in mean_volatility},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest