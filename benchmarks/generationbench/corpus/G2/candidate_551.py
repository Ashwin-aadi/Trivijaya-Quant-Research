from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: momentum (recent price "
        "trend) and volatility (price range over the last 20 days). The idea is that stocks "
        "with strong momentum but low recent volatility might be poised for a breakout."
    )

    def __init__(self, momentum_window: int = 10, volatility_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._volatility_window - 1)
        if closes.height < self._momentum_window + self._volatility_window - 1:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            price_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(price_series) < self._momentum_window + self._volatility_window - 1:
                continue

            # Calculate momentum score
            recent_returns = [(price_series[i] / price_series[i - 1] - 1.0) for i in range(1, self._momentum_window)]
            momentum_score = sum(recent_returns) / self._momentum_window
            momentum_scores[symbol] = momentum_score

            # Calculate volatility score (range of prices)
            ranges = [price_series[i + 1] - price_series[i] for i in range(len(price_series) - 1)]
            volatility_score = max(ranges[-self._volatility_window:]) / price_series[0]
            volatility_scores[symbol] = volatility_score

        # Combine scores
        combined_scores: dict[str, float] = {}
        for symbol in momentum_scores.keys():
            combined_scores[symbol] = momentum_scores[symbol] * (1 - volatility_scores[symbol])

        top_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest