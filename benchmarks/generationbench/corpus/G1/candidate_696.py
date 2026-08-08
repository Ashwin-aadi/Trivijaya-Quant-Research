from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects the top-performing stocks over a lookback period based on "
        "their cumulative returns. High momentum is expected to outperform in the near term."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns: pl.DataFrame = (
            history.lazy()
            .select(pl.col("symbol"), (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"))
            .group_by("symbol")
            .agg((pl.col("return").sum()).alias("cumulative_return"))
            .sort("cumulative_return", descending=True)
            .select(pl.col("symbol"), (pl.col("cumulative_return") / self._window).alias("normalized_return"))
        ).collect()

        top_symbols = [row["symbol"] for row in returns.to_dicts()[:5]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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