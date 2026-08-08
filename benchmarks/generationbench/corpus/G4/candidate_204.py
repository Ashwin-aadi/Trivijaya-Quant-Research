from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks with strong "
        "past performance and overweighting them in the portfolio. It aims to capture upward "
        "momentum of outperforming stocks while mitigating risk from underperformers through "
        "regular rebalancing."
    )

    def __init__(self, window: int = 180, top_n_percent: float = 0.3) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = (
            closes.select(pl.col("adj_close").rolling_sum().over("symbol"))
            .group_by("symbol")
            .agg((pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("cumulative_return"))
            .sort("cumulative_return", descending=True)
        )

        top_n = int(len(view.symbols) * self._top_n_percent)
        picks: list[str] = cumulative_returns.head(top_n)["symbol"].to_list()
        weights = {s: 1.0 / top_n for s in picks}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest