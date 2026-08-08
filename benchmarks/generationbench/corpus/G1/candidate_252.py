from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks that have performed well "
        "relative to the market in the recent past to continue outperforming. By ranking "
        "stocks based on their recent returns and allocating capital to the top performers, "
        "the strategy aims to capture this persistent performance pattern."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns_df = (
            closes
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .select(["symbol", "session_date", "return"])
        )

        # Rank symbols by returns
        ranked = (
            returns_df
            .group_by("symbol")
            .agg(
                (pl.col("return").mean().alias("average_return"))
            )
            .sort("average_return", descending=True)
            .select(["symbol", "average_return"])
        )

        top_symbols = [row.symbol for row in ranked.rows()][:self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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