from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "Leverage the tendency of stock prices to revert to their historical averages over a short period (1-5 trading days), aligning with mean reversion principles. This mechanism ensures a clear directional bias, making it robust and reliable."
    )

    def __init__(self, lookback: int = 20, threshold: float = 0.04, window_exit: int = 3) -> None:
        self._lookback = lookback
        self._threshold = threshold
        self._window_exit = window_exit

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]

        mean_reversion_scores: dict[str, float] = {}
        for symbol in symbols:
            close_prices = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            sma50 = sum(close_prices[-10:]) / 10
            deviations = [(close - sma50) / sma50 for close in close_prices]
            z_scores = [abs(dev) if dev > 0 else 0 for dev in deviations]

            recent_deviation = abs(z_scores[-1])
            mean_reversion_score = max(recent_deviation - self._threshold, 0)

            if mean_reversion_score >= self._threshold:
                mean_reversion_scores[symbol] = mean_reversion_score

        sorted_scores = sorted(mean_reversion_scores.items(), key=lambda x: x[1], reverse=True)
        selected_symbols = [symbol for symbol, score in sorted_scores][:20]
        weights = {s: 5.0 / len(selected_symbols) for s in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest