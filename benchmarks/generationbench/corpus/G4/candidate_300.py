from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum20d(Strategy):
    rationale = (
        "This strategy exploits the tendency for stocks with strong past performance to "
        "continue outperforming in the near future. By ranking stocks based on their "
        "recent returns and selecting top performers, we aim to capture ongoing momentum."
    )

    def __init__(self, window: int = 30, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Rank symbols by average return
        ranked = (
            history.sort("avg_return", descending=True)
            .select(["symbol", "avg_return"])
            .head(self._top_n)
        )

        # Convert to weights dictionary
        picks = [row["symbol"] for row in ranked.rows()]
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