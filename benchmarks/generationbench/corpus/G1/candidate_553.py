from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of stock prices to revert "
        "to their historical means. Identifying stocks that have deviated significantly "
        "from their moving averages can provide profitable trading opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            pl.col("adj_close").mean().alias("mean")
        )
        latest_close = view.closes().select(pl.exclude("session_date"))
        spread = (latest_close - mean_close["mean"]).rename("spread")

        # Filter out symbols with insufficient data
        symbols_with_data = history.select("symbol").distinct().to_dict()["symbol"]
        filtered_spread = spread.filter(spread["symbol"].is_in(symbols_with_data))
        if filtered_spread.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Identify top absolute spreads
        abs_spread = filtered_spread.with_columns(
            (pl.col("spread").abs()).alias("abs_spread")
        )
        top_abs_spreads = abs_spread.sort("abs_spread", descending=True).select(
            ["symbol"]
        )

        top_n_symbols = [row["symbol"] for row in top_abs_spreads.slice(0, 5)]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest