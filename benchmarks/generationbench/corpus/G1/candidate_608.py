from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Directional moves in price are often followed by a confirmation through volume. "
        "High volume during a directional move suggests strong momentum and can signal "
        "a continuation of the trend."
    )

    def __init__(self, window: int = 10, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Identify directional moves
        direction_moves = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_series = [float(v) for v in history[f"{symbol}.close"].to_list()]
            volume_series = [float(v) for v in history[f"{symbol}.volume"].to_list()]

            # Calculate daily returns and filter by threshold
            daily_returns = [
                (close / prev_close - 1.0)
                if prev_close != 0
                else 0.0
                for close, prev_close in zip(close_series[1:], close_series[:-1])
            ]
            high_volume_days = [volume > volume_series[-1] * self._threshold for volume in volume_series[1:]]

            # Check for directional moves with high volume confirmation
            move_start = 0
            for i, (return_val, vol_val) in enumerate(zip(daily_returns, high_volume_days)):
                if return_val > 0 and vol_val:
                    direction_moves.append((symbol, history["session_date"][i + 1]))
                    break

        # Select top N symbols with directional moves
        direction_moves.sort(key=lambda x: x[1], reverse=True)
        picks = [move[0] for move in direction_moves[: self._window]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest