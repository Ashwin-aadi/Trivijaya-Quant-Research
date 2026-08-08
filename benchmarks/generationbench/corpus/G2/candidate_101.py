from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakouts occur when a stock breaks out of its recent range but does not "
        "reverse course. This can indicate sustained momentum and potentially a continuation of "
        "the trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            latest_close = float(history.filter(pl.col("session_date") == stamp)["adj_close"][0])
            history_slice = (
                history.select(
                    pl.col("session_date"),
                    (pl.col("adj_close") - pl.col("open")).alias("range")
                )
                .filter((pl.col("symbol") == symbol) & (pl.col("session_date") < stamp))
                .sort("session_date", descending=False)
            )

            if history_slice.height < self._window:
                continue

            highest_high = float(history_slice.select(pl.max("high")).to_series()[0])
            lowest_low = float(history_slice.select(pl.min("low")).to_series()[0])

            if latest_close > highest_high and (latest_close - lowest_low) / (highest_high - lowest_low) >= 0.8:
                breakout_symbols.append(symbol)

        weights = {symbol: 1.0 for symbol in breakout_symbols}
        return Signal(
            information_available_at=stamp, weights={s: weight for s in weights.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest