from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only the most liquid stocks are considered for "
        "investment. Equal weighting across these stocks helps to distribute risk evenly and "
        "can improve portfolio performance by reducing concentration risks."
    )

    def __init__(self, liquidity_threshold: int = 10_000_000) -> None:
        self._liquidity_threshold = liquidity_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = tuple(set(view.symbols))
        volume_history = (
            history.filter(pl.col("symbol").is_in(symbols))
                   .group_by("symbol")
                   .agg((pl.col("volume").sum()).alias("total_volume"))
                   .sort("total_volume", descending=True)
                   .head(self._liquidity_threshold)
                   .select(["symbol"])
                   .to_dict(True)["symbol"]
        )

        if not volume_history:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volume_history)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in volume_history}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest