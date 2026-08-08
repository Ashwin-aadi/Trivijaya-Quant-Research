from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "This strategy aims to exploit mean reversion by buying stocks that have fallen below "
        "their trailing average price level over a certain period."
    )

    def __init__(self, window: int = 20, threshold: float = 0.95) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).mean()
        threshold_value = (1.0 - self._threshold) * mean_close

        symbol_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            if len(adj_closes) < self._window + 1:
                continue

            latest_close = adj_closes[-1]
            mean_adj_close = sum(adj_closes[1:]) / self._window
            if latest_close <= (mean_adj_close * threshold_value):
                symbol_signals[symbol] = 0.95  # Assign a significant weight for reversion

        total_weight = sum(symbol_signals.values())
        weights = {symbol: value / total_weight for symbol, value in symbol_signals.items()}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest