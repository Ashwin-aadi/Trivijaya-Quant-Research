from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout in the short-term trend, continuation of that trend is common. "
        "This strategy aims to identify such opportunities by looking for strong close prices "
        "that suggest continued movement in the direction of the breakout."
    )

    def __init__(self, window: int = 10, threshold: float = 0.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].to_list()[-self._window - 20:-20]]
            last_10_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].to_list()[-20:]]
            if len(closes) < self._window + 20 or len(last_10_closes) != 10:
                continue

            last_close = closes[-1]
            breakout_condition = any([c > (last_close * (1 + self._threshold)) for c in last_10_closes])
            if not breakout_condition:
                continue

            continuation_condition = all([(c - last_close) / last_close <= -self._threshold for c in last_10_closes])
            if continuation_condition:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest