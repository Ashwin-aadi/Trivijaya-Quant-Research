from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting assigns weights to stocks based on their liquidity "
        "measured by volume. This approach aims to balance risk and reward by considering the "
        "ease of trading for each stock."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history
            .group_by("symbol")
            .agg(pl.col("volume").sum().alias("total_volume"))
            .sort("total_volume", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if len(liquidity) < self._window:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest