from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are often more efficient in price discovery and trading. "
        "By equal-weighting these highly liquid stocks, the strategy aims to benefit from "
        "the efficiency of the market while diversifying risk across a subset of the most active "
        "stocks in the NIFTY 100 index."
    )

    def __init__(self, top_n: int = 5) -> None:
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_df = _calculate_liquidity(history)
        ranked_symbols = (
            liquidity_df.sort("liquidity", descending=True)["symbol"].to_list()[: self._top_n]
        )
        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_liquidity(history: pl.DataFrame) -> pl.DataFrame:
    volume_df = history.select(
        pl.col("symbol"), pl.col("volume").sum().alias("total_volume")
    )
    liquidity_df = (
        volume_df.with_columns(
            (pl.col("total_volume") / 1000000).cast(pl.Int64).alias("volume_million")
        )
        .group_by("symbol")
        .agg((pl.col("volume_million").sum().alias("total_volume_million"),))
    )

    return liquidity_df.sort("total_volume_million", descending=True)