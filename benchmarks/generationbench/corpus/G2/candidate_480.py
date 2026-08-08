from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that assets with the best recent performance "
        "are likely to continue outperforming in the near future. By focusing on these top performers, "
        "we can exploit this trend for potential returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            closes.melt().with_columns(
                (pl.col("value") / pl.col("value").shift(1) - 1.0).alias("return")
            )
        )

        # Select the top N performers based on recent cumulative returns
        top_n_symbols = _select_top_performers(returns, self._window)

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _select_top_performers(returns: pl.DataFrame, window: int) -> list[str]:
    # Calculate cumulative returns for the past window days
    cum_returns = (
        returns.group_by("symbol")
        .agg(
            (pl.col("return").sum().alias("cum_return"))
        )
        .sort("cum_return", descending=True)
        .select("symbol")
        .to_series()
        .to_list()[:window]
    )

    return cum_returns