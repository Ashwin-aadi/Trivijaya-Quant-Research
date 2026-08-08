from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion occurs when a stock's price deviates significantly from its long-term "
        "mean. This strategy aims to identify stocks that have moved away from their historical "
        "price levels and are likely to revert. By buying such stocks, we can capture the mean "
        "reverting effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_adj_close = (
            history.groupby("symbol").agg(pl.col("adj_close").mean().alias("m"))
        )
        latest_closes = view.closes()
        diffs: pl.DataFrame = (
            latest_closes.join(mean_adj_close, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") - pl.col("m")).abs().alias("diff").cast(pl.Float64)
            )
            .sort("diff", descending=True)
        )

        top_n_symbols = [s for s in diffs["symbol"].to_list()[: self._window]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest