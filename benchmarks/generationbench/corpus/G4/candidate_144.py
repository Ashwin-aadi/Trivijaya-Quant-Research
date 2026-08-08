from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum3m(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum in the Indian market by "
        "investing in stocks that have performed well over a recent period while selling or shorting underperformers. "
        "The economic mechanism is based on behavioral finance and investor psychology, where past performance tends to persist."
    )

    def __init__(self, lookback_days: int = 90, top_n: int = 20) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(self._lookback_days) - 1.0).alias("cumulative_return")
            )
            .group_by("symbol")
            .agg(pl.col("cumulative_return").mean().alias("avg_return"))
        )

        if cumulative_returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        sorted_returns = cumulative_returns.sort("avg_return", descending=True)
        top_symbols = [str(row["symbol"]) for row in sorted_returns.rows()][:self._top_n]

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