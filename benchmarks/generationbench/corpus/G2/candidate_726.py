from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers over a recent period are more likely to continue "
        "outperforming due to the momentum effect. This strategy aims to identify such stocks and "
        "allocate capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns for each symbol
        returns = (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0).alias("r")
        ranked_returns = (
            view.closes().with_columns(returns)
            .group_by("symbol")
            .agg((pl.col("r").mean().alias("avg_return")))
            .sort(pl.col("avg_return"), descending=True)
            .select("symbol", "avg_return")
        )

        if ranked_returns.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Select top symbols based on average returns
        top_symbols = [row["symbol"] for row in ranked_returns.to_dicts()[:5]]
        
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