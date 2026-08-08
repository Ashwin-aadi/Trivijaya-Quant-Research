from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the recent volatility of "
        "a stock and its momentum over a longer term. High volatility can indicate increased "
        "market interest, while positive momentum suggests a bullish outlook."
    )

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._long_window)

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volatility = ((pl.Series(close_prices).rolling_std(self._short_window))[-1]).item()
            volatilities[symbol] = volatility

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            momentum_score = (close_prices[-1] / close_prices[0]) - 1.0
            momentum_scores[symbol] = momentum_score

        combined_scores = {
            symbol: volatilities.get(symbol, 0) + momentum_scores.get(symbol, 0)
            for symbol in view.symbols
            if symbol in volatilities and symbol in momentum_scores
        }

        sorted_symbols = [
            s[0]
            for s in sorted(combined_scores.items(), key=lambda x: -x[1])
            if combined_scores[s[0]] > 0.5
       ][:3]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest