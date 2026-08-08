from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategies capitalize on the idea that after a stock has "
        "broken out of its previous price range, it often continues to move in the direction of "
        "the breakout. This can be measured by identifying stocks that have recently broken "
        "out and then showing further upward movement."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window - 1)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            hist_data = history.select(
                pl.col("session_date"), pl.col(symbol).alias("adj_close")
            )
            if hist_data.height < self._window + self._continuation_window - 1:
                continue

            breakout_price = (
                hist_data.tail(self._window)["adj_close"].to_list()[-1]
            )
            breakout_day = (
                hist_data.tail(self._window)
                .select(pl.col("session_date"))
                .tail(1)
                .item()
            )

            continuation_days = (
                hist_data.filter(
                    (pl.col("session_date") > breakout_day) &
                    (pl.col("adj_close") > breakout_price)
                )
                .sort("session_date")
                .select(pl.col("adj_close"))
                .head(self._continuation_window - 1)
            )

            if continuation_days.height >= self._continuation_window - 1:
                breakout_symbols.append(symbol)

        weights = {s: 1.0 / len(breakout_symbols) for s in breakout_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest