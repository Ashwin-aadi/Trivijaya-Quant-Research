from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are often more efficient in pricing and can provide a "
        "more stable return profile. By equal-weighting these highly liquid stocks, we aim to "
        "capitalize on the broader market's ability to quickly react to news and information."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(pl.col("volume").rolling_max(self._window).over("symbol") > 0)
            .group_by("symbol")
            .agg(
                pl.count().alias("count"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
        )

        if liquidity_screened.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_return = (
            liquidity_screened.lazy()
            .select(
                (pl.col("return").mean().alias("mean_return")),
                pl.col("count").max().alias("max_count"),
            )
            .collect()
        )

        filtered_symbols = mean_return.select(pl.col("symbol")).to_dict()["symbol"]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest