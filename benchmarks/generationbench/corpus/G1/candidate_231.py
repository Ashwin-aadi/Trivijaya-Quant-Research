from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key factor in market efficiency. Highly liquid stocks are more "
        "likely to be fairly priced and provide better execution for trades. This strategy "
        "equal-weights the most liquid NIFTY 100 constituents based on their trading volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_series = (
            history.filter(pl.col("symbol").is_in(view.symbols))
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
            .select("symbol")
            .head(self._window)["symbol"]
        )

        if volume_series.is_empty():
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(volume_series.to_list())
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in volume_series},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest