from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy combines the momentum of a stock with its volatility. High "
        "momentum stocks may continue to rise due to positive sentiment, while low "
        "volatility can indicate stable performance. By selecting symbols that score "
        "high on both metrics, we aim to capture stocks with strong but steady growth."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        volatility_scores: dict[str, float] = {}
        momentum_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            # Calculate 10-day volatility using median absolute deviation
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(prices) < self._volatility_window:
                continue
            mad_deviation = pl.Series(prices).median_abs_dev()
            volatility_scores[symbol] = 1 - (mad_deviation / max(prices))  # Normalize

            # Calculate 20-day momentum using percentage change from first close
            prices = [float(v) for v in closes[symbol].drop_nulls().to_list()[-self._momentum_window:]]
            if len(prices) < self._momentum_window:
                continue
            last_close, first_close = float(prices[-1]), float(prices[0])
            momentum_scores[symbol] = (last_close - first_close) / first_close

        # Combine scores into a single composite score
        combined_scores = {s: 0.5 * v + 0.5 * volatility_scores[s] for s in volatility_scores}
        sorted_symbols = sorted(combined_scores, key=lambda k: combined_scores[k], reverse=True)
        top_symbols = sorted_symbols[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest