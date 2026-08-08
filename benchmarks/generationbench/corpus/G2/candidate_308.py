from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong conviction among traders. "
        "A large volume increase alongside a price move suggests that the market is "
        "committing to a new trend direction, which can provide predictive power for future "
        "price movements."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            session_dates = [date.fromisoformat(d) for d in history["session_date"].to_list()]
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volume_values = [float(v) for v in history["volume"][symbol].drop_nulls().to_list()]

            if len(close_values) < self._window + 1:
                continue

            # Calculate the directional move
            last_close = close_values[-1]
            prev_close = close_values[-2]
            move = (last_close - prev_close) / abs(prev_close)

            # Check for volume increase in the direction of the price move
            if session_dates[-1] == view.as_of:
                current_volume = volume_values[-1]
                last_volume = volume_values[-2]

                if move > 0 and last_volume < current_volume:  # Upward move with increased volume
                    breakout_signals[symbol] = close_values[-1]
                elif move < 0 and last_volume > current_volume:  # Downward move with decreased volume

                    breakout_signals[symbol] = close_values[-1]

        if not breakout_signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_signals)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in breakout_signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest