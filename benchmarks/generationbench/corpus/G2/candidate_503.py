from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeMomentumVolatility(Strategy):
    rationale = (
        "Combining momentum and volatility characteristics can provide a more robust signal. "
        "High momentum stocks often see increased trading volume due to investor interest, while low volatility indicates stable price movements. "
        "By combining these signals, we aim to identify stocks with strong historical performance that are also relatively less volatile."
    )

    def __init__(self, momentum_window: int = 20, vol_window: int = 30) -> None:
        self._momentum_window = momentum_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._vol_window)

        if history.is_empty() or history.height < self._momentum_window + self._vol_window:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].to_list()]
            if len(adj_closes) < self._momentum_window + self._vol_window:
                continue

            # Momentum score: average daily return over the last 20 days
            daily_returns = [(adj_closes[i] - adj_closes[i - 1]) / adj_closes[i - 1]
                             for i in range(1, len(adj_closes))]
            momentum_score = sum(daily_returns[-self._momentum_window:]) / self._momentum_window

            # Volatility score: standard deviation of daily returns over the last 30 days
            volatility_score = (sum([(x - sum(daily_returns[-self._vol_window:]) / self._vol_window) ** 2 for x in daily_returns[-self._vol_window:]]) /
                                (self._vol_window - 1)) ** 0.5

            momentum_scores[symbol] = momentum_score
            volatility_scores[symbol] = volatility_score

        # Combine scores to rank stocks: higher momentum and lower volatility are better
        final_scores = {symbol: momentum_scores[symbol] / (1 + volatility_scores[symbol]) for symbol in momentum_scores}
        
        sorted_symbols = sorted(final_scores, key=lambda s: -final_scores[s])
        top_n = min(len(sorted_symbols), 5)  # Select the top N symbols based on final score

        weight = 1.0 / len(top_n)
        return Signal(information_available_at=stamp, weights={s: weight for s in sorted_symbols[:top_n]})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest