from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate that a significant number of market "
        "participants are in agreement about the direction of price action. This can lead to "
        "sustained trends and potential profit opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter out symbols without enough data
        symbols_with_data = [symbol for symbol in view.symbols if symbol in history.columns]

        # Initialize a dictionary to store volume-confirmed directional moves
        move_signals: dict[str, float] = {}

        for symbol in symbols_with_data:
            close_series = history[symbol]
            adj_close_values = [float(v) for v in close_series.drop_nulls().to_list()]

            if len(adj_close_values) < self._window + 1:
                continue

            # Calculate daily returns
            returns = [(adj_close_values[i] - adj_close_values[i-1]) / adj_close_values[i-1] if adj_close_values[i-1] != 0 else 0 for i in range(1, len(adj_close_values))]

            # Filter the last 20 days to find significant moves
            recent_returns = returns[-self._window:]

            # Check if there is a clear directional move with sufficient volume confirmation
            avg_volume = history[symbol].select(pl.col("volume").mean()).item()
            volume_confirmation = sum(1 for return_val in recent_returns if abs(return_val) > 0.05 and pl.col(symbol).filter(pl.col("session_date") < view.as_of).select(pl.col("volume")).sum() > avg_volume * 2)

            # Only consider symbols with significant directional moves
            if volume_confirmation / self._window >= 0.3:
                move_signals[symbol] = sum(recent_returns) / len(recent_returns)

        # Sort by the signal strength and pick top-performing symbols
        sorted_symbols = sorted(move_signals.items(), key=lambda x: abs(x[1]), reverse=True)
        selected_symbols = [symbol for symbol, _ in sorted_symbols[:5]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(selected_symbols)
        signal_weights = {s: weight_per_symbol for s in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest