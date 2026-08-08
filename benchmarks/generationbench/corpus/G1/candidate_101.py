from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After identifying a breakout candidate, we look for sustained price movement in the "
        "direction of the breakout. This suggests that the initial momentum is likely to continue."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_closes = view.closes(lookback=self._window)

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in breakout_closes.columns:
                continue
            values = [float(v) for v in breakout_closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            last_close = values[-1]
            max_high = max(values)
            if last_close >= max_high and last_close == max_high:
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in view.symbols:
            history_df = history.select(["session_date", f"{symbol}"])
            if symbol not in history.columns or history_df.height < self._continuation_window + 1:
                continue
            last_close = float(history_df["session_date"].max().item())
            values = [float(v) for v in history_df[symbol].to_list()]
            first_continuation_price = min(values[-self._continuation_window :])
            if last_close > max_high and first_continuation_price >= max_high:
                continuation_symbols.append(symbol)

        common_symbols = list(set(breakout_symbols).intersection(continuation_symbols))
        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(common_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in common_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest