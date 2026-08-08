from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reversion strategies rely on the idea that stocks which have recently deviated "
        "significantly from their historical price levels are likely to revert. This is based "
        "on the assumption that extreme deviations from a stock's mean price level are temporary."
    )

    def __init__(self, window: int = 20, deviation_multiplier: float = 1.5) -> None:
        self._window = window
        self._deviation_multiplier = deviation_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        symbols_with_history = set(latest_close.keys()) & set(history.columns)

        def calculate_z_score(symbol: str) -> float:
            symbol_data = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_close")
            )
            close_values = [float(v) for v in symbol_data[f"{symbol}_close"].to_list()]
            mean_close = sum(close_values) / len(close_values)
            std_dev_close = (sum((v - mean_close) ** 2 for v in close_values) / (
                len(close_values) - 1)) ** 0.5
            z_score = (latest_close[symbol] - mean_close) / std_dev_close if std_dev_close else 0
            return z_score

        z_scores = {symbol: calculate_z_score(symbol) for symbol in symbols_with_history}
        top_reverting_symbols = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        weights = {symb: self._deviation_multiplier / len(top_reverting_symbols) for symb, _ in top_reverting_symbols}

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items() if weight > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest