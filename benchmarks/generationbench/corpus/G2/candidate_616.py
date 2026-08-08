from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Trailing reversion exploits mean-reverting behavior in stock prices. "
        "If a stock's price has been consistently above its trailing average over the past 20 days, "
        "it may revert to that average, providing an entry point for buyers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate trailing average
        trailing_avg = (
            closes.lazy()
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("trailing_avg")))
            .collect()
        )

        # Join with current close prices to calculate difference from trailing avg
        diff_from_avg = (
            view.closes()
            .join(trailing_avg, on="symbol", how="left")
            .with_columns(
                (pl.col("adj_close") - pl.col("trailing_avg")).alias("diff_from_avg")
            )
            .sort("session_date", descending=False)
        )

        # Find symbols with the largest positive difference from trailing avg
        top_symbols = (
            diff_from_avg.filter(pl.col("diff_from_avg") > 0).tail(self._window)["symbol"]
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equally among the top symbols
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest