from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "This strategy combines the momentum of a stock with its volatility. High "
        "momentum stocks often outperform in the short term, while lower volatility can "
        "indicate stability and reduced risk."
    )

    def __init__(self, momentum_window: int = 10, volatility_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._volatility_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        momentum_scores: list[float] = []
        volatility_scores: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]

            if len(close_values) < self._momentum_window + self._volatility_window - 1:
                continue

            # Calculate momentum score as the percentage change over the window
            mom_change = (close_values[-1] - close_values[0]) / close_values[0]
            momentum_scores.append(mom_change)

            # Calculate volatility score as the standard deviation of returns
            rets = [(close_values[i] - close_values[i - 1]) / close_values[i - 1] for i in range(1, len(close_values))]
            vol_score = pl.Series(rets).std()
            volatility_scores.append(float(vol_score))

        top_momentum_indices = sorted(range(len(momentum_scores)), key=lambda i: momentum_scores[i], reverse=True)[:5]
        low_volatility_indices = sorted(range(len(volatility_scores)), key=lambda i: volatility_scores[i])[:3]

        selected_symbols = set()
        for index in top_momentum_indices:
            symbol = view.symbols[index]
            if symbol not in history.columns or any(history[symbol].is_null().sum() > 0):
                continue
            selected_symbols.add(symbol)

        for index in low_volatility_indices:
            symbol = view.symbols[index]
            if symbol not in history.columns or any(history[symbol].is_null().sum() > 0):
                continue
            selected_symbols.add(symbol)

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest