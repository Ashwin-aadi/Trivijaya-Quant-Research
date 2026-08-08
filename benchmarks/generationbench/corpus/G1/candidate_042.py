from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength against the broader market are more likely to outperform. "
        "This strategy identifies stocks that have performed better than their peers over a certain period."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]
        relative_strengths: list[float] = []

        for symbol in symbols:
            symbol_closes = history[symbol].to_list()
            universe_closes = history["adj_close"].to_list()

            if len(symbol_closes) < self._window or len(universe_closes) < self._window:
                continue

            # Calculate simple returns
            symbol_returns = [float(v / symbol_closes[i-1] - 1.0) for i, v in enumerate(symbol_closes)][self._window:]
            universe_returns = [float(v / universe_closes[i-1] - 1.0) for i, v in enumerate(universe_closes)][self._window:]

            # Calculate mean returns
            symbol_mean_return = sum(symbol_returns) / len(symbol_returns)
            universe_mean_return = sum(universe_returns) / len(universe_returns)

            # Calculate relative strength
            relative_strength = (symbol_mean_return - universe_mean_return) / universe_mean_return if universe_mean_return != 0 else 0.0

            relative_strengths.append(relative_strength)

        top_symbols = [symbols[i] for i in sorted(range(len(relative_strengths)), key=lambda j: relative_strengths[j], reverse=True)[:5]]

        weight = 1.0 / len(top_symbols)
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest