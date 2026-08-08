from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendVolatility(Strategy):
    rationale = (
        "This strategy seeks to capture trends by investing in symbols that have shown strong"
        " upward momentum over the past month while also having low historical volatility. "
        "The idea is that high momentum combined with low volatility may indicate a robust and "
        "consistent trend, potentially offering higher returns with less risk."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 60) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._momentum_window + self._volatility_window)

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue

            adj_closes = [float(v) for v in history[symbol]["adj_close"].to_list()]
            if len(adj_closes) < self._momentum_window + 1:
                continue

            # Calculate momentum score
            price_changes = [
                (adj_closes[i] - adj_closes[i - 1]) / adj_closes[i - 1]
                for i in range(1, self._momentum_window)
            ]
            mean_momentum = sum(price_changes) / len(price_changes)
            momentum_scores[symbol] = mean_momentum

            # Calculate volatility score
            prices = [float(v) for v in history[symbol]["adj_close"].to_list()[-self._volatility_window:]]
            vol = (pl.Series(prices).std())**2
            volatility_scores[symbol] = 1 / vol if vol > 0 else float("inf")

        # Select top N symbols based on composite score
        momentum_sorted = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        volatility_sorted = sorted(volatility_scores.items(), key=lambda x: x[1])

        combined_scores: list[str] = []
        for symbol, _ in momentum_sorted:
            if symbol not in [x[0] for x in volatility_sorted]:
                combined_scores.append(symbol)

        top_symbols = combined_scores[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest