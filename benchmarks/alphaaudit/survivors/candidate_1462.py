from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonal trends in the Indian equity market can provide predictive signals. "
        "By identifying stocks that historically perform well during certain months or seasons, "
        "we can construct a strategy to capitalize on these trends."
    )

    def __init__(self, window: int = 12, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_close = sum(values) / self._window
            max_close = max(values)
            if (max_close - mean_close) / mean_close > self._threshold:
                seasonal_signals[symbol] = 1.0

        if not seasonal_signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(seasonal_signals.values())
        weighted_signals = {k: v / total_weight for k, v in seasonal_signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=weighted_signals,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest