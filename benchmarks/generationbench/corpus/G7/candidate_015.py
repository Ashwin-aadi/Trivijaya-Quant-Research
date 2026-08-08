from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy focuses on stocks with the highest daily volume over a 252-day lookback period, "
        "ensuring exposure to the most liquid securities. Equal weighting simplifies management and reduces costs."
    )

    def __init__(self, window: int = 252, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volume_df = history.select(
            pl.col("symbol"),
            pl.col("adj_close").last().alias("close"),
            pl.col("volume").sum().alias("total_volume"),
        )

        ranked_symbols = (
            volume_df.sort(pl.col("total_volume"), descending=True)
            .head(self._top_n)
            .select("symbol")
            .to_series()
            .to_list()
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(ranked_symbols)

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol for symbol in ranked_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest