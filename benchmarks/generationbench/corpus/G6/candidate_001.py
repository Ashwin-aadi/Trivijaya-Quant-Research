from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "The strategy focuses on stocks reverting to their historical means over shorter "
        "time horizons (days or weeks), capitalizing on both extreme price movements and "
        "their subsequent reversion."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 3.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            data = closes[[symbol]].drop_nulls()
            prices = [float(v) for v in data[symbol].to_list()]
            ma_20 = sum(prices[-self._window:]) / self._window
            std_dev = (sum((p - ma_20) ** 2 for p in prices[-self._window:]) / self._window) ** 0.5
            z_score = (prices[-1] - ma_20) / std_dev if std_dev > 0 else 0

            if z_score >= self._z_score_threshold:
                signals.append(symbol)
        signals = signals[:50]
        weight = 1.0 / len(signals) if signals else 0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest