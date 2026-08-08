from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies capitalize on the tendency of stock prices to revert "
        "to their mean over time. This strategy identifies stocks that have deviated significantly "
        "from their 20-day moving average and bets on a mean reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = pl.DataFrame(
            {"symbol": view.symbols, "recent_close": [float(v) for v in view.closes().to_dict("records")]}
        )

        def rolling_mean(history: pl.DataFrame) -> float:
            return history["adj_close"].mean().item()

        mean_rets = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").rolling_mean(self._window).alias(f"mean_{self._window}d")),
                (pl.col("adj_close") - pl.col(f"mean_{self._window}d")).abs().max().alias("max_deviation"),
            )
            .collect()
        )

        symbols_with_deviation = mean_rets.select(["symbol", "max_deviation"])
        if symbols_with_deviation.height == 0:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = (
            symbols_with_deviation.sort("max_deviation", descending=True)
            .head(5)["symbol"]
            .to_list()
        )

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest