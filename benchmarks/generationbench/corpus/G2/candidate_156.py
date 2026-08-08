from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the tendency for assets that have "
        "performed well in recent periods to continue outperforming over a short horizon. "
        "This strategy ranks stocks by their performance and allocates capital to the top performers."
    )

    def __init__(self, lookback_window: int = 30) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window or len(closes.columns) == 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple returns
        returns = (
            view.history().select(
                pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            ).with_columns(pl.col("return").is_nan().alias("missing"))
        )
        
        # Filter out symbols with missing data
        if returns.select(pl.sum("missing")).item() > 0:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = (
            closes.join(returns.group_by("symbol").agg(pl.col("return").mean().alias("average_return")), on="symbol", how="inner")
                .with_columns(
                    (pl.col("average_return") / pl.col("average_return").max() * 10).cast(pl.Int32()).alias("ranking")
                )
                .sort("ranking", descending=True)
        )

        top_symbols = symbol_returns["symbol"].to_list()[:5]
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