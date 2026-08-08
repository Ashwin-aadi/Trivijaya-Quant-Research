from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices tend to revert to an average over time. "
        "In this strategy, we identify stocks whose recent returns are extreme compared to their "
        "historical mean, indicating a potential future mean-reverting move."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = set(view.symbols).intersection(set(history["symbol"].to_list()))
        relevant_history = history.filter(pl.col("symbol").is_in(symbols))

        mean_close = (
            relevant_history.groupby("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("return_mean")
            )
            .select("symbol", "return_mean")
        )

        current_closes = view.closes(lookback=None)
        mean_return = (
            relevant_history.groupby("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("return")
            )
            .join(mean_close, on="symbol", how="inner")
            .select(pl.col("return") - pl.col("return_mean"))
        )

        sorted_symbols = mean_return.sort(
            "return",
            descending=False,
        )["symbol"].to_list()[:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest