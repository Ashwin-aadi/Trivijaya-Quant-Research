from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum30d(Strategy):
    rationale = (
        "Cross-sectional momentum exploits recent price trends across stocks to identify "
        "those showing strong outperformance for future returns. This strategy focuses on "
        "stocks with the highest positive cumulative returns over 30 days."
    )

    def __init__(self, window: int = 30, top_n: int = 25) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate cumulative returns for each symbol
        closes = view.closes()
        price_changes = (closes[closes.columns[1:]] / closes[closes.columns[0]].shift(1) - 1).fillna(0)
        cum_returns = price_changes.cumsum()

        # Filter out symbols with low market capitalization
        mcap_filter = history["adj_close"] > 1_000_000_000
        filtered_history = history.filter(mcap_filter)

        if filtered_history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols based on cumulative returns
        ranked_symbols = filtered_history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("avg_close"),
            cum_returns.sum().alias("cum_return")
        ).sort("cum_return", descending=True)

        if ranked_symbols.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in ranked_symbols.rows()[:self._top_n]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest