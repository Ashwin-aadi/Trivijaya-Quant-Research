from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "recently to continue performing well in the near future. This strategy aims to allocate"
        " capital towards the top performers from a recent period."
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
        returns = closes.clone()
        for symbol in view.symbols:
            if symbol not in returns.columns or returns[symbol].is_empty():
                continue
            returns = returns.with_columns(
                (pl.col(symbol) / pl.col(symbol).shift(1) - 1.0).alias(f"{symbol}_return")
            )

        # Rank symbols by return over the window period
        ranked_returns = (
            returns
            .group_by("session_date")
            .agg(pl.col(s for s in view.symbols if f"{s}_return" in returns.columns)
                 .mean().rank(descending=True, method="dense").alias("rank"))
            .sort("session_date", descending=False)
        )

        # Identify top performers
        latest_ranks = ranked_returns.filter(pl.col("session_date") == stamp).select(
            "symbol", "rank"
        ).to_dict(as_series=False)

        picks: list[str] = [k for k, v in latest_ranks["rank"].items() if v <= self._top_n]

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