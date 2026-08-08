from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity can be an indicator of liquidity traders' confidence in the market. "
        "Highly liquid stocks are less likely to experience extreme price movements, and "
        "they tend to have more stable trading volumes. By equal weighting these stocks, "
        "we aim to benefit from this stability."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_closes = (
            history.filter(pl.col("volume").quantile(0.9) > 0)
                    .group_by("symbol")
                    .agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
                    .sort("r", descending=True)
                    .head(self._window)[["symbol"]]
        )

        symbols = liquidity_screened_closes.get_column("symbol").to_list()
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest