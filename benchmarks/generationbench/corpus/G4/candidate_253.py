from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy capitalizes on short-horizon mean reversion in the Indian market by "
        "identifying stocks that have deviated significantly from their historical price range."
    )

    def __init__(self, lookback: int = 30, threshold_high: float = 2.0, threshold_low: float = -2.0) -> None:
        self._lookback = lookback
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            price_series = [float(v) for v in closes[symbol].to_list()]
            high_prices = [v for _, v in zip(price_series, price_series)]
            low_prices = [v for v in price_series]
            average_price_change = sum(p2 - p1 for p1, p2 in zip(price_series[:-1], price_series[1:])) / (len(price_series) - 1)
            volatility = ((max(high_prices) - min(low_prices)) / average_price_change if average_price_change != 0 else 0)

            recent_closes = [float(v) for v in view.closes(lookback=self._lookback)[symbol].to_list()[-self._lookback:]]
            recent_prices_mean = sum(recent_closes) / self._lookback
            z_score = (recent_closes[-1] - recent_prices_mean) / volatility if volatility != 0 else 0

            mean_reversion_scores[symbol] = abs(z_score)

        sorted_symbols = sorted(mean_reversion_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top_high_symbols = [symbol for symbol, score in sorted_symbols[:20]]
        bottom_low_symbols = [symbol for symbol, score in sorted_symbols[-20:]]

        if not (top_high_symbols or bottom_low_symbols):
            return Signal(information_available_at=stamp, weights={})

        long_positions = {s: 1.0 / len(bottom_low_symbols) for s in bottom_low_symbols}
        short_positions = {s: -1.0 / len(top_high_symbols) for s in top_high_symbols}

        combined_weights = {**long_positions, **short_positions}
        return Signal(information_available_at=stamp, weights=combined_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest