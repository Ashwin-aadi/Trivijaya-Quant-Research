from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often seen as less risky and can provide more stable returns. "
        "By tilting our portfolio towards these low-volatility stocks, we aim to reduce overall risk."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .select(["symbol", "volatility"])
        )

        top_n = daily_returns.sort("volatility").tail(self._lookback_period)["symbol"].to_list()
        if not top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest