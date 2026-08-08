from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum identifies the top performers across symbols and allocates "
        "capital to these leaders, leveraging historical price data."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = history.lazy().group_by("symbol").agg(
            (pl.col("adj_close").mean() / pl.col("adj_close").shift(1).mean() - 1.0)
            .alias("momentum")
        ).collect().sort("momentum", descending=True)

        if closes.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in closes.rows()[:self._top_n]]
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