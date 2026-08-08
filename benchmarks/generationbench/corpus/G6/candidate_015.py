from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class HighLiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy identifies and invests in a diversified portfolio of high-liquidity stocks "
        "screened based on daily trading volume exceeding INR 5 million. The portfolio is equally "
        "weighted to mitigate risks associated with illiquid securities."
    )

    def __init__(self, lookback: int = 30, min_volume: float = 5_000_000) -> None:
        self._lookback = lookback
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Filter stocks based on daily trading volume
        volume_df = (
            history.filter(pl.col("volume") > self._min_volume)
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").last() / pl.col("adj_close").shift(1) - 1.0).alias("r"),
                (pl.col("volume")).alias("volume"),
            )
        )

        # Sort by daily return and volume
        sorted_df = (
            volume_df.sort(
                ["volume", "r"], descending=[True, False]
            ).select(["symbol"])
            .head(100)
        )

        symbols = [row[0] for row in sorted_df.to_dict(as_series=False).values()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest