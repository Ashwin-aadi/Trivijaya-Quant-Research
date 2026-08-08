from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining a momentum indicator with a volatility measure can provide a more robust "
        "signal for entry. Momentum helps identify strong trends, while volatility indicates "
        "market instability or strength."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 30) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._momentum_window, self._volatility_window))
        if closes.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._momentum_window:
                continue

            # Calculate momentum score (simple moving average)
            sma = sum(values[-self._momentum_window:]) / self._momentum_window
            momentum_scores[symbol] = values[-1] - sma

            # Calculate volatility score (standard deviation of returns)
            returns = [v2 / v1 - 1.0 for v1, v2 in zip(values[:-1], values[1:])]
            volatility_scores[symbol] = pl.DataFrame({"returns": returns}).select(
                (pl.col("returns").std()).alias("volatility")
            ).height

        combined_scores = {
            symbol: momentum_scores[symbol] * 0.5 + volatility_scores.get(symbol, 0) * 0.5
            for symbol in view.symbols
        }

        top_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest