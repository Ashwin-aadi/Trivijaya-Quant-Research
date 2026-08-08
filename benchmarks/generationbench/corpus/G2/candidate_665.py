from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "A breakout followed by a strong continuation can signal a change in trend and potential "
        "momentum. This strategy seeks to identify such breakouts where the price continues to rise, "
        "indicating sustained buying pressure."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            history_df = history.select(["session_date", "adj_close"]).filter(
                pl.col("symbol") == symbol
            )
            if history_df.height < self._window + 1:
                continue

            adj_closes = [float(v) for v in history_df["adj_close"].to_list()]
            max_close = max(adj_closes[-self._window :])
            breakout_day = adj_closes.index(max_close)
            if breakout_day == len(adj_closes) - 1 or breakout_day < self._window:
                continue

            post_breakout = adj_closes[breakout_day + 1 :]
            if all(c > max_close for c in post_breakout):
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