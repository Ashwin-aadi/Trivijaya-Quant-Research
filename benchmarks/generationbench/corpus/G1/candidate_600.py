from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed in "
        "the recent past to continue outperforming. This strategy ranks assets based on their "
        "returns over a short window and invests in the top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).sort("session_date")

        # Aggregate to get latest return for each symbol
        latest_returns = (
            history.select(["symbol", "session_date", "return"])
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Get top N symbols based on average returns
        top_symbols = (
            latest_returns.sort("avg_return", descending=True)
            .head(self._window)
            .select(["symbol"])
            .to_dict(as_series=False)["symbol"]
        )

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