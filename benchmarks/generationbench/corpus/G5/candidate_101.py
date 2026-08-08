from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout candidate, we look for sustained price movement in the "
        "direction of the breakout to confirm its validity. This strategy aims to enter positions "
        "in symbols that continue their post-breakout trend."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or len(closes[symbol].to_list()) < self._continuation_window + 1:
                continue

            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            breakout_day_close = values[-2]
            post_breakout_values = values[-self._continuation_window - 1:-1]

            if len(post_breakout_values) < self._continuation_window:
                continue

            trend_direction = (post_breakout_values[0] > breakout_day_close) * 2 - 1
            valid_trend = all(((trend_direction == 1 and value >= breakout_day_close) or
                               (trend_direction == -1 and value <= breakout_day_close))
                              for value in post_breakout_values)

            if valid_trend:
                picks.append(symbol)

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest