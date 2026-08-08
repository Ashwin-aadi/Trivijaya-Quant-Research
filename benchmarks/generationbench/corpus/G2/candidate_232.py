from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reversion is a common phenomenon where asset prices tend to return to their "
        "historical means. By identifying assets that have deviated significantly from their "
        "trailing mean, we can exploit this tendency for potential returns."
    )

    def __init__(self, window: int = 60, k: float = 1.5) -> None:
        self._window = window
        self._k = k

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_adj_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean_adj_close")
        )
        latest_closes = view.closes().select(["session_date", *view.symbols])

        merged = (
            history.join(mean_adj_close, on="symbol", how="inner")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_adj_close")).alias("deviation"),
                ((pl.col("adj_close") / pl.col("mean_adj_close")) - 1).alias("ratio"),
            )
            .sort(["symbol", "session_date"], descending=False)
            .filter((pl.col("deviation").abs() > self._k * pl.col("mean_adj_close")))
        )

        symbols = merged["symbol"].to_list()
        weights: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in latest_closes.columns:
                continue
            weight = 1.0 / len(symbols)
            weights[symbol] = weight

        return Signal(
            information_available_at=stamp,
            weights={s: weights.get(s, 0) for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest