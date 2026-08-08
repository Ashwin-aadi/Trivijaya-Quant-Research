from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy exploits the liquidity premium by selecting a diversified portfolio "
        "of equally weighted stocks from highly liquid Indian equities. The higher trading volume "
        "of these stocks suggests lower transaction costs and market inefficiencies, potentially "
        "resulting in higher returns for investors willing to engage in such trades."
    )

    def __init__(self, window: int = 30, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_data = (
            history.select(pl.col("symbol"), pl.col("volume"))
                   .group_by("symbol")
                   .agg(pl.col("volume").mean().alias("avg_volume"))
                   .sort("avg_volume", descending=True)
                   .head(self._top_n)
        )

        if volume_data.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks = [row["symbol"] for row in volume_data.to_dicts()]
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