from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies capitalize on the phenomenon where a stock that has "
        "broken out from its recent range is more likely to continue moving in the direction of"
        "the breakout. This strategy identifies stocks that have recently broken out and then "
        "looks for further continuation of the price movement."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_history = history.select(pl.col("symbol"), pl.col("session_date"), pl.col("close"))
            latest_close = float(recent_history.filter(pl.col("symbol") == symbol)[-1]["close"])
            breakout_price = float(
                recent_history.with_columns((pl.col("close").shift(-self._lookback)).alias("breakout_price"))[
                    -2
                ]["breakout_price"]
            )
            if latest_close > breakout_price:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest