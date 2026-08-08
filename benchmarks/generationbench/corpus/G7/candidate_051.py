from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that highly traded stocks are selected, reducing the "
        "risk of large trades impacting the stock price. Equal weighting across the selected "
        "stocks provides a simple and effective way to diversify risk."
    )

    def __init__(self, window: int = 30, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        volume_screened = (
            history.group_by("symbol")
                   .agg(pl.col("volume").sum().alias("total_volume"))
                   .sort("total_volume", descending=True)
                   .head(self._top_n)
        )
        symbols = volume_screened["symbol"].to_list()

        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest