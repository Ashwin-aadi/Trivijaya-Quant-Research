from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility often precedes price reversals or continuation trends. By identifying "
        "high-volatility stocks and following their trend, we can capture significant movements."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            closes = [float(v) for v in history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol))["adj_close"].to_list()]
            if len(closes) < self._window:
                continue

            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            mean_return = sum(returns) / len(returns)
            volatility = (sum([abs(r - mean_return)**2 for r in returns]) / len(returns)) ** 0.5
            score = volatility / abs(mean_return)

            if score > self._threshold:
                volatility_scores[symbol] = score

        top_symbols = sorted(volatility_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest