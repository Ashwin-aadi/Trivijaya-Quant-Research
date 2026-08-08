from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in stocks that have performed better than the broader market "
        "can provide excess returns. This strategy selects the top performers based on their "
        "10-day cumulative return relative to the NIFTY 50 index."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        nifty_closes = closes.filter(pl.col("symbol") == "NIFTY50")
        if nifty_closes.height < 1:
            return Signal(information_available_at=stamp, weights={})

        nifty_last_close = float(nifty_closes.select(pl.col("adj_close").last()).item())
        nifty_returns = (closes["adj_close"] / nifty_closes["adj_close"].shift(1) - 1.0).alias("nifty_return")

        symbols = [symbol for symbol in view.symbols if symbol != "NIFTY50"]
        performance = history.filter(pl.col("symbol").is_in(symbols)).with_columns(
            (pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).alias("cumulative_return")
        )

        ranked_performance = (
            performance.group_by("symbol")
            .agg(nifty_returns.sum().alias("nifty_total_return"), pl.col("cumulative_return").mean().alias("avg_return"))
            .sort(pl.col("nifty_total_return", "avg_return"), descending=[True, True])
            .head(self._window)
        )

        top_symbols = ranked_performance.select("symbol").to_list()
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