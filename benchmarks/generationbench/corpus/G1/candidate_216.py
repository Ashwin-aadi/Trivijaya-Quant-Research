from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for recent winners (in terms of returns) "
        "to continue outperforming in the near future. This strategy allocates capital to the top "
        "performers over a short period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        returns = (
            closes.sort("session_date")
            .select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r")
            )
            .group_by("symbol")
            .agg(pl.col("r").mean().alias("avg_return"))
            .sort("avg_return", descending=True)
        )

        if returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [row["symbol"] for row in returns.head(self._top_n).to_dicts()]
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