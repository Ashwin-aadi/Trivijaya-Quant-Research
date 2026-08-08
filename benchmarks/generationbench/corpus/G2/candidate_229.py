from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion theory suggests that stock prices and returns eventually move back toward "
        "a long-term mean or average. In a short horizon, extreme deviations from this mean are likely to be corrected."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history[symbol].drop_nulls().to_list()
            close_prices = [float(v) for v in symbol_history[-self._window :]]
            mean_close_price = sum(close_prices) / len(close_prices)
            latest_close_price = float(history[symbol][-1])
            z_score = (latest_close_price - mean_close_price) / max(
                0.01, abs(mean_close_price)
            )
            if abs(z_score) > 2:  # Consider only extreme cases for reversion
                mean_reversion_signals[symbol] = z_score

        if not mean_reversion_signals:
            return Signal(information_available_at=stamp, weights={})

        target_allocation = 1.0 / len(mean_reversion_signals)
        weighted_signal = {s: abs(z) * target_allocation for s, z in mean_reversion_signals.items()}
        return Signal(
            information_available_at=stamp,
            weights={k: v if v > 0 else 0.0 for k, v in weighted_signal.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest