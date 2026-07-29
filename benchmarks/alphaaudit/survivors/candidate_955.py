from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of reduced volatility. During such periods, "
        "stocks may be more likely to breakout in the near future. By identifying symbols with "
        "compressed ranges, we can potentially capture these breakouts."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        range_compression = (
            history.group_by("symbol")
            .agg(
                (pl.col("high").min() - pl.col("low").max()).alias("range_diff"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
            )
            .sort("range_diff", descending=False)
        )

        if range_compression.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in range_compression.to_dicts()]
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