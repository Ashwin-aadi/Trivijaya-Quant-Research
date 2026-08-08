from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a proxy for market efficiency and price discovery. Higher liquidity"
        " suggests that the asset has more active trading and is less likely to be overvalued."
        " By equal-weighting assets based on their liquidity, we can capture the benefits of"
        " high liquidity while maintaining a diversified portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate liquidity as the average daily trading volume over the window
        liquidity = (
            history.groupby("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .with_columns((pl.col("avg_volume") / pl.col("avg_volume").sum()).alias("weight"))
        )

        # Filter out symbols with no historical data or zero volume to avoid division by zero
        liquidity = liquidity.filter(
            (pl.col("avg_volume") > 0) & pl.col("symbol").is_in(view.symbols)
        )
        
        if liquidity.height == 0:
            return Signal(information_available_at=stamp, weights={})

        # Equal-weight the remaining symbols based on their liquidity
        weight = 1.0 / liquidity.height
        weights = {row["symbol"]: float(row["weight"]) * weight for _, row in liquidity.iter_rows()}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest