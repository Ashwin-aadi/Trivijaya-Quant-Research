from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength20d(Strategy):
    rationale = (
        "This strategy exploits the 'momentum effect' by selecting stocks with recent "
        "outperformance against the broader market index. Stocks showing strong relative "
        "strength are longed while others are held in cash."
    )

    def __init__(self, window: int = 20, top_n_percentage: float = 0.2) -> None:
        self._window = window
        self._top_n_percentage = top_n_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        cumulative_returns = (
            history.select(pl.col("symbol"), pl.col("close").rolling_sum())
                   .with_columns(
                       (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("cumulative_return")
                   )
        ).sort("session_date", descending=True)

        ranked_stocks = cumulative_returns.sort("cumulative_return", descending=True)
        top_n_count = int(len(view.symbols) * self._top_n_percentage)
        picks: list[str] = [row["symbol"] for row in ranked_stocks.head(top_n_count).to_dicts()]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest