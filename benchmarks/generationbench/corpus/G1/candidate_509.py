from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "If a stock breaks out of its recent trading range and continues to move in the "
        "direction of the breakout, it often suggests sustained momentum. This strategy aims "
        "to identify such stocks for potential long-term gains."
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
            if symbol not in history["symbol"].to_list():
                continue
            recent_highs = [float(v) for v in history.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("session_date") >= date(2020, 1, 1))
                ["high"].drop_nulls().to_list()
            )]
            recent_lows = [float(v) for v in history.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("session_date") >= date(2020, 1, 1))
                ["low"].drop_nulls().to_list()
            )]

            if len(recent_highs) < self._window or len(recent_lows) < self._window:
                continue

            breakout_high = max(recent_highs[-self._window:])
            breakout_low = min(recent_lows[-self._window:])

            recent_close = float(history.filter(
                (pl.col("symbol") == symbol)
                & (pl.col("session_date") == stamp)
            )["adj_close"].to_list()[0])
            if (
                recent_close > breakout_high
                and history.filter(
                    (pl.col("symbol") == symbol)
                    & ((pl.col("high") > pl.col("low")) & (pl.col("high") >= breakout_high))
                    & (pl.col("session_date") < stamp)
                ).height < self._window - 1
            ):
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