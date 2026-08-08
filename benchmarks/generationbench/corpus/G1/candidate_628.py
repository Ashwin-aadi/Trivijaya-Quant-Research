from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Reversion to the mean is a fundamental principle in technical analysis. "
        "Prices that have moved too far from their historical average are likely to revert. "
        "This strategy aims to identify such cases and generate trades."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean"))
                   .with_columns((pl.col("adj_close") - pl.col("mean")).alias("deviation"))
        )

        recent_closes = view.closes(lookback=self._window).select(symbols)
        deviations = [float(recent_closes[symbol].to_list()[-1]) for symbol in symbols]

        filtered_symbols = [
            symbol
            for deviation, symbol in zip(deviations, symbols)
            if abs(deviation) >= 2 * mean_close.filter(pl.col("symbol") == symbol)["deviation"].item()
        ]

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest