from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in the direction of the breakout after a certain period. "
        "This strategy identifies stocks that have recently broken out and then trended further in "
        "that direction."
    )

    def __init__(self, window: int = 20, continuation_lookback: int = 10) -> None:
        self._window = window
        self._continuation_lookback = continuation_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_lookback)

        if history.height < self._window + self._continuation_lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].drop_nulls().to_list()]
            if len(adj_closes) < self._window + self._continuation_lookback:
                continue

            breakout_day = adj_closes.index(max(adj_closes[-self._window:]))
            trend_direction = 1.0 if adj_closes[breakout_day] < adj_closes[
                breakout_day - 1
            ] else -1.0
            post_breakout_trend = [trend_direction * (v - adj_closes[breakout_day]) for v in
                                   adj_closes[breakout_day:breakout_day + self._continuation_lookback]]

            if all(post_breakout_trend) > 0:
                breakout_symbols.append(symbol)

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