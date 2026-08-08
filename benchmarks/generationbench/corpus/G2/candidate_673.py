from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion is a theory that asset prices and investment returns eventually move "
        "back toward the long-term mean or average. In an equity market context, stocks that "
        "have performed poorly recently are expected to outperform in the short term as their "
        "prices revert to historical norms."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean_close")
        )
        latest_closes = view.closes().select(pl.exclude("session_date"))

        # Calculate the difference between current close and historical mean
        diff_from_mean = (
            latest_closes.join(mean_close, on="symbol", how="left")
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_close")).alias("diff_from_mean")
            )
            .select(["symbol", "diff_from_mean"])
        )

        # Identify symbols with the largest negative deviations
        sorted_diff = diff_from_mean.sort("diff_from_mean").filter(pl.col("diff_from_mean") < 0)
        top_n_symbols = [row[0] for row in sorted_diff.head(self._window).to_dict(orient="records")]

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