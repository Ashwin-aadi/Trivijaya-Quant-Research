from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often continue in direction of initial move. This strategy "
        "identifies symbols that have recently broken out and then holds them for a period to "
        "capitalize on potential continuation."
    )

    def __init__(self, breakout_window: int = 20, holding_period: int = 10) -> None:
        self._breakout_window = breakout_window
        self._holding_period = holding_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._breakout_window + self._holding_period)

        if history.is_empty() or history.height < self._breakout_window + self._holding_period:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            low_window_min = min(adj_closes[-self._breakout_window:])
            high_window_max = max(adj_closes[-self._breakout_window:])
            latest_close = adj_closes[-1]

            if latest_close > high_window_max or latest_close < low_window_min:
                breakout_symbols.append(symbol)

        weights = {s: 0.5 for s in breakout_symbols[:4]} if breakout_symbols else {}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest