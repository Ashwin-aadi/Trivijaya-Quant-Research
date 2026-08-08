from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for market confidence. Higher liquidity often indicates "
        "greater willingness to trade and thus potentially higher future returns. This strategy "
        "screens stocks by their average daily trading volume over the last 20 days, then equally "
        "weights them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_vol = (
            history.group_by("symbol")
            .agg((pl.col("volume").sum() / pl.col("volume").count()).alias("avg_volume"))
            .sort("avg_volume", descending=True)
        )

        if avg_vol.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        filtered_symbols = (
            avg_vol.filter(pl.col("avg_volume") > 0).select("symbol").to_list()[0]
        )
        selected_symbols = [s for s in symbols if s in filtered_symbols][: self._window]

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest