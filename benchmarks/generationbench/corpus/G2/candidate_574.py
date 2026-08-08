from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality effects suggest that certain times of the year may be more favorable for "
        "equity returns due to predictable corporate activities or market sentiments. By identifying "
        "historical patterns, we can exploit these periods."
    )

    def __init__(self, seasonal_window: int = 20, top_n: int = 5) -> None:
        self._seasonal_window = seasonal_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._seasonal_window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_history = history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("avg_adj_close")
        )

        # Determine the top N performing symbols based on their average adj close
        sorted_symbols = (
            symbol_history.sort("avg_adj_close", descending=True)
            .head(self._top_n)
            .select(["symbol"])
            .to_dict(False)
        )
        picks = [s["symbol"] for s in sorted_symbols]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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