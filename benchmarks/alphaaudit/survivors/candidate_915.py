from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting ensures that stocks with higher liquidity "
        "receive more weight in the portfolio. This strategy aims to leverage the trading "
        "activity of liquid assets to potentially benefit from their price movements."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .select(["symbol", "total_volume"])
        )

        symbols = [row["symbol"] for row in liquidity.to_dicts()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest