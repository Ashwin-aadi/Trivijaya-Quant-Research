from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation patterns. After a significant price move, "
        "the market may continue in the same direction due to惯性 and psychological factors. "
        "Identifying stocks that have recently broken out and showing continued strength can be "
        "a profitable strategy."
    )

    def __init__(self, window: int = 20, continuation_window: int = 5) -> None:
        self._window = window
        self._continuation_window = continuation_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_window)
        if history.height < self._window + self._continuation_window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            if len(adj_closes) < self._window + self._continuation_window:
                continue

            # Find the breakout point
            for i in range(self._window, len(adj_closes)):
                if adj_closes[i] >= max(adj_closes[:self._window]):
                    break
            else:
                continue  # No breakout found within the window

            # Check continuation pattern over next `continuation_window` days
            post_breakout = [adj_closes[i + j] for j in range(1, self._continuation_window + 1)]
            if all(post_breakout[i] > post_breakout[i - 1] for i in range(len(post_breakout))):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:self._continuation_window]
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