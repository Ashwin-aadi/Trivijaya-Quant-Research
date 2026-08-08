from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying symbols where the current price "
        "is far from their trailing average, we can generate buy or sell signals based on "
        "this principle."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        trailing_mean = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("trailing_mean"))
            .with_columns((pl.col("adj_close") / pl.col("trailing_mean") - 1.0).alias("z_score"))
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in trailing_mean.columns:
                continue
            z_score = float(trailing_mean[trailing_mean["symbol"] == symbol]["z_score"].to_list()[0])
            if abs(z_score) > 1.5:
                picks.append(symbol)

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