from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "After a strong breakout, the stock may continue its trend. "
        "Identifying such continuation patterns can provide profitable opportunities."
    )

    def __init__(self, window: int = 20, breakout_threshold: float = 1.5) -> None:
        self._window = window
        self._breakout_threshold = breakout_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            hist = history[symbol]
            adj_close_vals = [float(v) for v in hist.drop_nulls().to_list()]
            recent_close = adj_close_vals[-1]
            breakout_price = max(adj_close_vals[:-self._window])
            if recent_close > self._breakout_threshold * breakout_price:
                breakout_symbols.append(symbol)

        continuation_symbols: list[str] = []
        for symbol in breakout_symbols:
            hist = history[symbol]
            adj_close_vals = [float(v) for v in hist.drop_nulls().to_list()]
            if len(adj_close_vals) < self._window + 1:
                continue
            trend_slope = (adj_close_vals[-1] - adj_close_vals[0]) / self._window
            if trend_slope > 0.0:
                continuation_symbols.append(symbol)

        weights: dict[str, float] = {}
        for symbol in continuation_symbols[:5]:
            weights[symbol] = 1.0 / len(continuation_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in continuation_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest