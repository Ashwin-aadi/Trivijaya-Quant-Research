from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining the 20-day moving average cross and the relative strength index (RSI) "
        "provides a more robust entry signal by capturing both trend strength and overbought/oversold conditions."
    )

    def __init__(self, ma_window: int = 20, rsi_window: int = 14, threshold: float = 70.0) -> None:
        self._ma_window = ma_window
        self._rsi_window = rsi_window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._ma_window + self._rsi_window)
        if closes.height < self._ma_window + self._rsi_window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            # Calculate 20-day moving average
            ma_20 = (
                closes[symbol]
                .drop_nulls()
                .to_list()[-self._ma_window:]
                .sum()
                / self._ma_window
            )

            # Calculate RSI
            price_changes = [
                float(c - o) for c, o in zip(
                    closes[symbol].drop_nulls().to_list()[1:],
                    closes[symbol].drop_nulls().to_list()[:-1]
                )
            ]
            gains = [p if p > 0 else 0 for p in price_changes]
            losses = [-l if l < 0 else 0 for l in price_changes]

            avg_gain = sum(gains) / self._rsi_window
            avg_loss = abs(sum(losses)) / self._rsi_window

            rs = 10.0 * (avg_gain / avg_loss) if avg_loss > 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))

            # Check for overbought/oversold condition
            if rsi >= self._threshold:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest