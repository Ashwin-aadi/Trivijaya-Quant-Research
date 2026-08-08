from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength outperform the market over time. "
        "This strategy seeks to identify such stocks by comparing their recent performance against the broader universe."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.height < self._window or closes.width < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        prices = closes.drop("session_date")
        universe_mean_returns = (prices / prices.shift(1).fill_null(1.0) - 1.0).mean(
            skip_nulls=True
        )
        relative_strengths = (
            (closes["adj_close"] / closes["adj_close"].shift(1).fill_null(1.0) - 1.0)
            .to_list()
            + [float(universe_mean_returns)]
        )

        sorted_indices = list(
            reversed(
                pl.DataFrame(relative_strengths, schema=["strength"])
                .with_column((pl.col("strength").rank(method="ordinal", descending=True)).alias("rank"))
                .select("symbol")
                .sort("rank")
                .iter_rows()
                .map(lambda row: row["symbol"])
            )
        )

        if not sorted_indices:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = sorted_indices[:5]
        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest