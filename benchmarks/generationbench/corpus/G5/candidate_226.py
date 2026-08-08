from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data suggests that certain stocks exhibit higher returns during specific times of the year."
        "By identifying these seasonal patterns, we can allocate our portfolio to perform better in those periods."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            daily_ret = (
                history.filter(pl.col("symbol") == symbol)
                       .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
                       .select(["session_date", "return"])
                       .sort(by="session_date")
            )
            if daily_ret.height < self._window:
                continue
            avg_returns[symbol] = float(daily_ret.select(pl.col("return").mean()).item())

        sorted_symbols = [
            s for s, r in sorted(avg_returns.items(), key=lambda item: -abs(item[1]))
        ][: self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest