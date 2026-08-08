from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks that have outperformed the NIFTY 100 index over a "
        "recent lookback period. Stocks with higher relative strength are expected to "
        "continue their positive performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(view.symbols) + 1 < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_100_history = view.closes(lookback=self._window).select(
            pl.col(view.as_of - date.timedelta(days=self._window)).alias("nifty_close")
        )

        stock_returns = (
            history.select(pl.col("symbol").alias("symbol"))
                   .join(history.select(pl.col(["symbol", "adj_close"])), on="symbol")
                   .with_column((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
        )
        
        nifty_returns = (
            history.select(pl.col(view.as_of - date.timedelta(days=self._window)).alias("nifty_100_close"))
                   .join(nifty_100_history, on="symbol")
                   .with_column(
                       (pl.col("adj_close") / pl.col("nifty_close") - 1.0).alias("relative_return"),
                       pl.col("return").alias("stock_return")
                   )
        )

        mean_relative_return = nifty_returns.groupby("symbol").agg(
            (pl.col("relative_return").mean()).alias("mean_rel_ret")
        ).sort("mean_rel_ret", descending=True).select("symbol")

        if mean_relative_return.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in mean_relative_return.to_dicts()[:self._window]]
        
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest