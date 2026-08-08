from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks with relative weakness by entering when their RSI falls below 30. "
        "It uses a conservative approach for entry and multiple exit strategies to manage risk."
    )

    def __init__(self, window: int = 14, threshold_entry: float = 30, threshold_exit_high: float = 70) -> None:
        self._window = window
        self._threshold_entry = threshold_entry
        self._threshold_exit_high = threshold_exit_exit_high

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        rsi_values = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            gains = [(close_series[i] - close_series[max(0, i - 1)]) / (i + 1)
                     for i in range(len(close_series))]
            losses = [-gains[i] if g < 0 else 0 for i, g in enumerate(gains)]
            avg_gain = sum([max(g, 0) for g in gains[-self._window:]]) / self._window
            avg_loss = sum([abs(l) for l in losses[-self._window:]]) / self._window

            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append((symbol, rsi))

        if not rsi_values:
            return Signal(information_available_at=stamp, weights={})

        sorted_rsis = sorted(rsi_values, key=lambda x: x[1])
        top_n_symbols = [s for s, r in sorted_rsis if r < self._threshold_entry][:25]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest