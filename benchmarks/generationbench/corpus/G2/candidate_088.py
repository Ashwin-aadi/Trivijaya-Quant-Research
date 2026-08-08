from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reversion is a classical trading strategy where prices are expected to revert "
        "to the mean after extreme movements. This can be observed in equity markets when "
        "prices overshoot their historical average levels."
    )

    def __init__(self, window: int = 60, trailing_window: int = 30) -> None:
        self._window = window
        self._trailing_window = trailing_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._trailing_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        recent_closes = view.closes().drop_nulls().to_dict(as_series=False)
        trailing_means: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.columns:
                continue
            adj_close_values = [float(v) for v in history[symbol].to_list()]
            trailing_mean = sum(adj_close_values[-self._trailing_window:]) / self._trailing_window
            trailing_means[symbol] = trailing_mean

        signals: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in recent_closes or symbol not in trailing_means:
                continue
            current_price = recent_closes[symbol]
            trailing_mean = trailing_means[symbol]
            z_score = (current_price - trailing_mean) / max(abs(current_price - trailing_mean), 1e-8)
            if abs(z_score) > 2:  # Consider only extreme deviations for reversion
                signals[symbol] = -0.5 * z_score

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        normalized_weights = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=normalized_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest