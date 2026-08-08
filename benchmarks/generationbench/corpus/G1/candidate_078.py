from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks that outperform the broader market can provide a consistent "
        "source of alpha. This strategy ranks stocks based on their recent performance relative to "
        "the NIFTY 100 index and selects the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * 200:
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes(lookback=self._window).select(
            pl.col(view.symbols[0]).alias("nifty_close")
        )
        stock_closes = view.closes(lookback=self._window)

        returns_nifty = (
            (history.select(pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0))
            .group_by("session_date")
            .agg(pl.col("close").mean().alias("return"))
            .select(["session_date", "return"])
        )
        returns_stocks = (
            (stock_closes.join(returns_nifty, on="session_date", how="inner").drop(
                ["symbol"]
            ).select(
                pl.col(view.symbols).to_series() / pl.col("nifty_close") - 1.0
            ))
            .group_by("symbol")
            .agg(pl.Series.mean().alias("return_ratio"))
        )

        ranked = returns_stocks.sort("return_ratio", descending=True)
        picks: list[str] = [row.symbol for row in ranked.head(self._top_n).rows()]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest