from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform the market over long periods. "
        "By tilting our portfolio towards these stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["symbol"].unique()) < 5:
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").std().alias("volatility")))
                   .sort("volatility", descending=False)
                   .select(["symbol"])
                   .head(5)["symbol"]
                   .to_list()
        )

        if not low_vol_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_vol_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest