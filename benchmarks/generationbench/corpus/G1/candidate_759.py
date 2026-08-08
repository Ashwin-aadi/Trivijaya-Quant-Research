from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity is a critical factor in determining the ease of trading stocks. "
        "High liquidity stocks are more attractive as they can be traded without significantly "
        "affecting their price. This strategy selects the top 10 most liquid stocks for equal weighting."
    )

    def __init__(self, top_n: int = 10) -> None:
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=20)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily volume for each symbol
        daily_volumes = (
            history.group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
        )

        if daily_volumes.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        # Select top N symbols based on liquidity
        picks = [row[0] for row in daily_volumes.head(self._top_n).to_dict(False)]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest