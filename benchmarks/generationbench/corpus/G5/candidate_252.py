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
            .select(["symbol", "return"])
        )

        # Rank symbols by return
        ranked = returns_df.group_by("symbol").agg(
            (pl.col("return").mean().alias("average_return")).rank(method="dense", descending=True)
        ).sort("average_return")

        top_symbols: list[str] = [row["symbol"] for _, row in ranked.iter_rows() if row["average_return"] <= self._top_n]

        # Ensure at least one symbol is selected
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Assign weights based on rank and average return
        weight_sum = sum([1.0 / (rank + 1) for _, row in ranked.iter_rows() if row["average_return"] <= self._top_n])
        weights = {symbol: 1.0 / (rank + 1) / weight_sum for symbol, rank in zip(top_symbols, range(len(top_symbols)))}

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest