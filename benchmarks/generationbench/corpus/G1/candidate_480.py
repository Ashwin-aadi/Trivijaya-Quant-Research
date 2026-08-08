from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the recent past to continue outperforming the market. By weighting recent "
        "high performers, we aim to capture this effect."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        rank_df = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("momentum"),
            )
            .sort("momentum", descending=True)
            .head(self._top_n)
        )

        symbols = rank_df["symbol"].to_list()
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest