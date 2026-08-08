from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "By combining a momentum signal with a mean reversion signal, we aim to capture "
        "both trending and mean-reverting behavior in the market. Momentum signals work well "
        "in trending markets, while mean reversion is effective when prices return to their "
        "historical average."
    )

    def __init__(self, momentum_window: int = 20, mean_reversion_window: int = 60) -> None:
        self._momentum_window = momentum_window
        self._mean_reversion_window = mean_reversion_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._mean_reversion_window)
        if closes.height < self._mean_reversion_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_signal = self._calculate_momentum(view, self._momentum_window)
        mean_reversion_signal = self._calculate_mean_reversion(closes)

        combined_weights = {}
        for symbol in view.symbols:
            if symbol not in momentum_signal.weights or symbol not in mean_reversion_signal.weights:
                continue
            weight = (
                float(momentum_signal.weights[symbol])
                + float(mean_reversion_signal.weights[symbol])
            )
            combined_weights[symbol] = weight

        # Normalize weights to ensure they sum up to 1.0, with the remainder as cash.
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            for symbol in combined_weights:
                combined_weights[symbol] /= total_weight
        else:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in combined_weights.items()},
        )

    def _calculate_momentum(self, view: MarketView, window: int) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            latest_close = float(view.latest_close()[symbol])
            close_change = (latest_close - values[-1]) / values[-1]
            momentum_scores[symbol] = close_change

        sorted_symbols = [
            symbol for _, symbol in sorted(momentum_scores.items(), key=lambda item: abs(item[1]))
        ]
        top_symbols = sorted_symbols[:5]

        weight = 0.2
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols})

    def _calculate_mean_reversion(self, closes: pl.DataFrame) -> Signal:
        stamp = _latest_visible(view)
        if closes.height < self._mean_reversion_window:
            return Signal(information_available_at=stamp, weights={})

        mean_prices = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._mean_reversion_window:
                continue
            mean_price = float(sum(values[-self._mean_reversion_window :]) / self._mean_reversion_window)
            latest_close = float(view.latest_close()[symbol])
            deviation = (latest_close - mean_price) / mean_price
            mean_prices[symbol] = deviation

        sorted_symbols = [
            symbol for _, symbol in sorted(mean_prices.items(), key=lambda item: abs(item[1]))
        ]
        bottom_symbols = sorted_symbols[:5]

        weight = 0.2
        return Signal(information_available_at=stamp, weights={s: -weight for s in bottom_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest