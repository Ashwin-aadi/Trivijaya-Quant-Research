from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "High-liquidity stocks are often less impacted by market events and may exhibit more "
        "consistent performance. By equal-weighting stocks based on their liquidity, we aim to "
        "reduce the risk of holding less liquid assets that could be more volatile."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.lazy()
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") - pl.col("open")).abs().mean().alias("price_spread"),
            )
            .collect()
            .sort("total_volume", descending=True)
        )

        if liquidity.height < 10:
            return Signal(information_available_at=stamp, weights={})

        top_liquidity = [row["symbol"] for row in liquidity.to_dicts()[:10]]

        weight = 1.0 / len(top_liquidity)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_liquidity},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest