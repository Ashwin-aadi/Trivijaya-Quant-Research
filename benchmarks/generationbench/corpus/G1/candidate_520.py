from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels revert to the mean over time. By identifying assets that have deviated "
        "significantly from their trailing average price, we can find potential reversion "
        "opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("trailing_mean"))
            .with_columns(pl.col("trailing_mean") / pl.col("adj_close").shift(1) - 1.0)
            .filter(pl.col("trailing_mean").is_not_null())
            .select(
                [
                    "symbol",
                    (pl.col("trailing_mean") / history["adj_close"].mean()).alias(
                        "reversion_score"
                    )
                ]
            )
        )

        if mean_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            mean_close.sort("reversion_score", descending=True)
            .head(self._window)
            .select(["symbol"])
            .to_dict(as_series=False)["symbol"]
        )

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest