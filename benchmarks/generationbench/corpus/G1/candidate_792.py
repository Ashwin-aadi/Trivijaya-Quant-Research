from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the past to continue performing well in the future. This strategy buys top-performing "
        "stocks based on recent returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .head(self._window)
        )

        # Calculate mean return for each symbol
        mean_returns = (
            history.groupby("symbol").agg(pl.col("return").mean().alias("mean_return"))
            .sort("mean_return", descending=True)
            .select(["symbol", "mean_return"])
        )

        if mean_returns.height < self._top_n:
            top_symbols = [row["symbol"] for row in mean_returns.rows()]
        else:
            top_symbols = [row["symbol"] for row in mean_returns.head(self._top_n).rows()]

        # Allocate weights to the top symbols
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