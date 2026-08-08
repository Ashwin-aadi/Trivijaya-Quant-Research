from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts that continue to new highs after a period of consolidation indicate strong "
        "momentum and buying pressure. This strategy identifies such breakouts for potential "
        "long positions."
    )

    def __init__(self, window: int = 20, continuation_window: int = 10) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)

        if history.is_empty() or history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            adj_closes = history[symbol]["adj_close"].drop_nulls().to_list()
            if len(adj_closes) < self._window + self._continuation_window:
                continue

            # Find the breakout day
            for i in range(self._window, len(adj_closes)):
                if (
                    adj_closes[i] > max(adj_closes[:self._window])
                    and all(
                        adj_closes[j] >= adj_closes[i]
                        for j in range(i + 1, min(len(adj_closes), i + self._continuation_window))
                    )
                ):
                    breakout_symbols.append(symbol)
                    break

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