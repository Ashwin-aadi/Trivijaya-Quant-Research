from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify trending stocks by normalizing "
        "price changes with their historical volatility. During periods of low volatility, "
        "the price action may signal a continuation or reversal in the trend."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue

            # Calculate the price changes and their z-score
            changes = [
                (values[i] - values[i - 1]) / abs(values[i - 1])
                for i in range(1, len(values))
            ]
            mean_change = sum(changes) / len(changes)
            std_deviation = (
                sum((change - mean_change) ** 2 for change in changes) / (len(changes) - 1)
            ) ** 0.5

            # Z-score calculation
            z_score = (changes[-1] - mean_change) / std_deviation if std_deviation else 0
            if abs(z_score) >= self._z_score_threshold:
                symbol_trends[symbol] = changes[-1]

        selected_symbols = [
            s for s, t in sorted(symbol_trends.items(), key=lambda x: abs(x[1]), reverse=True)
        ][:5]
        weight = 1.0 / len(selected_symbols) if selected_symbols else 0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest