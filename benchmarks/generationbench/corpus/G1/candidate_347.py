from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: 20-day closing price momentum and "
        "50-day moving average of volume. A combination of these signals aims to identify stocks with both "
        "strong recent performance and increasing trading activity."
    )

    def __init__(self, window_close: int = 20, window_vol: int = 50) -> None:
        self._window_close = window_close
        self._window_vol = window_vol

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_close + self._window_vol)
        if history.height < self._window_close + self._window_vol:
            return Signal(information_available_at=stamp, weights={})

        closes = history["close"]
        vol_history = view.history(lookback=self._window_vol)["volume"]

        if closes.height < self._window_close or vol_history.height < self._window_vol:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: list[float] = []
        volume_scores: list[float] = []

        for symbol in view.symbols:
            close_values = [float(v) for v in closes[symbol].to_list()]
            vol_values = [float(v) for v in vol_history[symbol].drop_nulls().to_list()]

            if len(close_values) < self._window_close or len(vol_values) < self._window_vol:
                continue

            close_momentum_score = (close_values[-1] - close_values[0]) / sum(close_values)
            volume_mean = sum(vol_values) / len(vol_values)

            momentum_scores.append(close_momentum_score)
            volume_scores.append(volume_mean)

        symbols_with_high_momentum_and_volume: list[str] = [
            symbol
            for i, symbol in enumerate(view.symbols)
            if momentum_scores[i] >= 0.5 and volume_scores[i] > pl.col("volume").mean().item()
        ]

        if not symbols_with_high_momentum_and_volume:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_with_high_momentum_and_volume)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in symbols_with_high_momentum_and_volume
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest