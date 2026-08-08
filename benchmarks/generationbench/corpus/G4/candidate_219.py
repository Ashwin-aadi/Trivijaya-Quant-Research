from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidSmallCap(Strategy):
    rationale = (
        "This strategy focuses on exploiting the persistent profitability of small-cap stocks by "
        "screening for liquid small-cap stocks and applying an equal-weighted approach to capture excess returns. "
        "Higher-ranked stocks based on recent monthly returns are equally weighted in the portfolio at regular intervals."
    )

    def __init__(self, window: int = 60, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_filter = (history.select(pl.col("volume").sum())
                            .with_columns((pl.col("symbol") / pl.col("volume").sum()).alias("liquidity_ratio"))
                            .filter(pl.col("volume").mean() > 100_000)
                            .select("symbol"))

        if liquidity_filter.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [s for s in view.symbols if s in liquidity_filter.to_series().to_list()]
        closes = view.closes(lookback=self._window).select(pl.col(symbol_list))

        returns = (closes
                   .lazy()
                   .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
                   .collect()
                   .select(pl.col("symbol"), "return")
                   )

        monthly_returns = (returns
                           .group_by("symbol")
                           .agg([pl.col("return").mean().alias("monthly_return")])
                           .sort("monthly_return", descending=True)
                           .limit(self._top_n)
                           .select("symbol"))

        symbols = [s for s in monthly_returns["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest