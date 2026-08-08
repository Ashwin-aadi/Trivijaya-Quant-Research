from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion in financial markets suggests that asset prices and earnings will "
        "tend to move back towards the long-term mean or average level over time. Short-horizon "
        "mean reversion strategies look for stocks that have deviated significantly from their "
        "historical means and are expected to revert to those means."
    )

    def __init__(self, window: int = 10, deviation_threshold: float = 2.0) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores = {}
        for symbol in view.symbols:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_close")
            )
            if symbol_history.is_empty():
                continue

            close_prices = [float(v) for v in symbol_history[f"{symbol}_close"].to_list()]
            mean_price = float(pl.DataFrame(close_prices).mean().item())
            latest_close = view.latest_close()[symbol]
            deviation = (latest_close - mean_price) / mean_price
            if abs(deviation) > self._deviation_threshold:
                mean_reversion_scores[symbol] = deviation

        # Filter out symbols that do not meet the threshold
        mean_reversion_scores = {k: v for k, v in mean_reversion_scores.items() if abs(v) > self._deviation_threshold}

        if not mean_reversion_scores:
            return Signal(information_available_at=stamp, weights={})

        # Calculate equal-weighted positions based on the reversion scores
        weights = {}
        total_score = sum(abs(score) for score in mean_reversion_scores.values())
        for symbol, score in mean_reversion_scores.items():
            weight = 1.0 / len(mean_reversion_scores)
            if score > 0:
                weights[symbol] = -weight
            else:
                weights[symbol] = weight

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest