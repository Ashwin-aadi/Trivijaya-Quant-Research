from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks with strong recent "
        "performance to continue outperforming over short periods. By investing in the top "
        "performers, we aim to capture this momentum effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_performance = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").last() / pl.col("adj_close").first() - 1.0).alias("return"),
            )
            .sort("return", descending=True)
            .select(["symbol"])
        )

        top_symbols = symbol_performance.head(5)["symbol"].to_list()
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