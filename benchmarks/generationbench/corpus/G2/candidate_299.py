from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy aims to combine two characteristics: momentum and low volume. "
        "Momentum suggests that stocks which have recently performed well are likely to continue performing well in the near future. Low volume can indicate underfollowed or smaller companies, potentially offering higher volatility and thus more opportunities for profit."
    )

    def __init__(self, momentum_window: int = 10, low_volume_threshold: float = 5e6) -> None:
        self._momentum_window = momentum_window
        self._low_volume_threshold = low_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + 1)
        if history.is_empty() or history.height < self._momentum_window + 2:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        momentum_scores = []
        volume_scores = []

        for symbol in symbols:
            close_series = history[symbol].drop_nulls().to_list()
            high_close_ratio = max(close / close_series[0] for close in close_series[-self._momentum_window:])
            momentum_scores.append(high_close_ratio)

            volume_series = [float(v) for v in history[f"{symbol}_volume"].drop_nulls().to_list()]
            low_volume_score = 1 if any(volume < self._low_volume_threshold for volume in volume_series) else 0
            volume_scores.append(low_volume_score)

        combined_scores = [(momentum, volume) for momentum, volume in zip(momentum_scores, volume_scores)]
        selected_symbols = [symbol for symbol, (momentum, volume) in zip(symbols, combined_scores)
                            if momentum >= max(momentum_scores) and volume == 1]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest