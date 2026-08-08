from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks with strong past returns "
        "to continue performing well. This strategy ranks assets based on their return over a "
        "recent lookback period and allocates capital to top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        grouped = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window - 1) - 1.0).alias("return"),
            )
            .sort("return", descending=True)
        )

        if grouped.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [symbol for symbol in view.symbols if symbol in grouped["symbol"].to_list()]
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