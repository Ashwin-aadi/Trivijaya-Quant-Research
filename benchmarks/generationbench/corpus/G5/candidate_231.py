from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key factor in market efficiency. High liquidity stocks are more "
        "likely to be accurately priced and less prone to price manipulation. This strategy "
        "selects the top N most liquid stocks based on average daily trading volume over a "
        "lookback period and assigns equal weights to them."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily trading volume for each symbol
        volumes = (
            history.lazy()
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
        )

        if volumes.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in volumes.to_pandas().head(self._top_n).values]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_pandas().iloc[0][0]
    assert isinstance(newest, date)
    return newest