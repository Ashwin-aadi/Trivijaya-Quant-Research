from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to mean levels over time due to the tendency of markets to correct "
        "extreme price movements. By identifying symbols that are significantly above their "
        "trailing average prices, we can exploit this reversion tendency for potential gains."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_prices = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("avg_price")))
            .select(["symbol", "avg_price"])
        )
        latest_closes = view.closes()
        
        merged = latest_closes.join(
            avg_prices, on="symbol", how="inner"
        ).with_columns(
            (pl.col("adj_close") / pl.col("avg_price")).alias("reversion_factor")
        )

        top_symbols = (
            merged.sort("reversion_factor").select(["symbol"])
            .head(5)["symbol"].to_list()
        )
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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