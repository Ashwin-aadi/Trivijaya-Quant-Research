from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with strong recent relative performance "
        "are likely to continue outperforming the market. This strategy exploits this tendency by "
        "investing in the top-performing stocks over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative performance
        relative_performance = (
            (history["close"] / history["adj_close"].shift(self._window) - 1.0).alias("r")
        )
        perf_df = history.with_columns(relative_performance)
        
        # Identify top performers
        top_symbols = perf_df.sort("r", descending=True)["symbol"].to_list()[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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