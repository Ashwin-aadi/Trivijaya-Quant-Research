from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy leverages a composite of two characteristics: recent momentum and "
        "volume growth. A stock with high volume and increasing price trend is likely to "
        "continue its upward trajectory."
    )

    def __init__(self, momentum_window: int = 20, volume_threshold: float = 1.1) -> None:
        self._momentum_window = momentum_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + 1)
        if history.height < self._momentum_window + 2:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if (
                symbol not in history.columns
                or len(history[symbol].drop_nulls().to_list()) < self._momentum_window + 1
            ):
                continue

            close_series = [float(v) for v in history["close"][symbol].drop_nulls().to_list()]
            volume_series = [float(v) for v in history["volume"][symbol].drop_nulls().to_list()]

            if len(close_series) < self._momentum_window + 1 or len(volume_series) < self._momentum_window + 1:
                continue

            # Check for momentum
            close_change = (close_series[-1] - close_series[0]) / close_series[0]
            momentum_signal = close_change > 0.05

            # Check for volume growth
            volume_ratio = volume_series[-1] / volume_series[-2]
            volume_signal = volume_ratio >= self._volume_threshold

            if momentum_signal and volume_signal:
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
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