from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "The strategy aims to identify stocks that exhibit both strong momentum and low volatility. "
        "Strong momentum suggests a bullish trend, while low volatility indicates reduced risk of price fluctuation."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.height < max(self._momentum_window, self._volatility_window):
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_values) < self._momentum_window:
                continue
            momentum_score = (close_values[-1] - min(close_values)) / max(close_values)
            momentum_scores[symbol] = momentum_score

        volatility_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(close_values) < self._volatility_window:
                continue
            volatility_score = pl.Series(close_values).std()
            volatility_scores[symbol] = volatility_score

        final_scores: dict[str, float] = {}
        for symbol in momentum_scores.keys():
            if symbol not in volatility_scores:
                continue
            final_score = (momentum_scores[symbol] + 1) * (1 - volatility_scores[symbol])
            final_scores[symbol] = final_score

        sorted_symbols = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest