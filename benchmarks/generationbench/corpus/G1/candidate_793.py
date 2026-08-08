from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that if a stock's price has deviated significantly from its "
        "historical average, it is likely to return to that mean. This strategy identifies "
        "such deviations and bets against the recent trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("m"))
            .with_columns((pl.col("adj_close") - pl.col("m")).abs().alias("deviation"))
        )
        
        recent_closes = view.closes(lookback=self._window)
        if any(col.is_null().any() for col in recent_closes.columns):
            return Signal(information_available_at=stamp, weights={})

        symbols_with_deviation = (
            mean_close
            .join(recent_closes, on="symbol", how="inner")
            .select(pl.col("deviation").gt(1.0 * pl.col("m")).alias("exceeds_mean"))
            .filter(pl.col("exceeds_mean") == True)
        )

        if symbols_with_deviation.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks = [symbol for symbol in symbols_with_deviation["symbol"].to_list()]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: -weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest