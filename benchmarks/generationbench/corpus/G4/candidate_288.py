from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum6m(Strategy):
    rationale = (
        "This strategy exploits the momentum effect by selecting stocks with positive "
        "past performance and excluding underperformers. It aims to capture excess returns "
        "by maintaining a diversified portfolio of top-performing stocks."
    )

    def __init__(self, window: int = 180, top_n: int = 25) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 6-month trailing returns
        history = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(2)
            .select(pl.col("symbol"), "return")
        )

        # Rank stocks based on returns
        ranked = history.with_column(
            (pl.col("return").rank(method="dense", descending=True)).alias("rank")
        )

        # Select top N% for long positions and bottom N% for short positions
        symbols = [row["symbol"] for row in ranked.sort("rank").head(self._top_n).to_dicts()]
        weights: dict[str, float] = {s: 1.0 / self._top_n for s in symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest