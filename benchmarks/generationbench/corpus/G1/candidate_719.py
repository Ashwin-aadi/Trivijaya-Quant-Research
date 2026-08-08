from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy combines the recent price momentum with the volume trend to identify "
        "stocks that are both strong in terms of price and experiencing increased buying interest."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volumes = history["volume"].to_list()

        momentum_score: dict[str, float] = {}
        volume_trend: dict[str, int] = {}

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in closes]
            vol_values = [int(v) for v in volumes]

            if len(close_values) < self._window or len(vol_values) < self._window:
                continue

            # Calculate momentum score as the percentage change from the first to last closing price
            momentum_change = (close_values[-1] - close_values[0]) / close_values[0]
            momentum_score[symbol] = momentum_change

            # Calculate volume trend by counting upward changes in volume over the window period
            vol_trend_count = sum(1 for i in range(1, self._window) if vol_values[i] > vol_values[i - 1])
            volume_trend[symbol] = vol_trend_count

        # Combine scores to get a composite score
        composite_score = {
            symbol: (momentum_score.get(symbol, 0.0) + volume_trend.get(symbol, 0)) / 2 for symbol in view.symbols
        }

        sorted_symbols = sorted(composite_score.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest