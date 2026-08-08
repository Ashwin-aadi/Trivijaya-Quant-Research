from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "High volume directional moves are often indicative of institutional or large trader "
        "behaviour. These actors may have significant information that can lead to profitable "
        "trades if the move is confirmed by continued volume on the follow-up session."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            hist = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            close_series = [float(v) for v in hist["close"].drop_nulls().to_list()]
            volume_series = [int(v) for v in hist[symbol].get_column("volume").to_list()]

            if len(close_series) < self._window + 1:
                continue

            # Calculate the directional move
            move_direction = close_series[-1] - close_series[0]
            if move_direction == 0:
                continue

            # Check for volume confirmation on the follow-up session
            last_session_close = close_series[-1]
            last_session_volume = volume_series[-1]
            prev_session_close = close_series[-2]

            if (
                last_session_volume > max(volume_series[:-1])
                and abs(last_session_close - prev_session_close) / move_direction >= 0.5
            ):
                signals[symbol] = (last_session_close - prev_session_close) / move_direction

        # Normalize the signal scores to sum up to 1 if there are any
        total_score = sum(signals.values())
        if total_score > 0:
            for symbol in signals:
                signals[symbol] /= total_score

        return Signal(
            information_available_at=stamp, weights={s: weight for s, weight in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest