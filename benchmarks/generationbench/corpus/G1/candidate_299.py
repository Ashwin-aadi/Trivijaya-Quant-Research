from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines a breakout condition with a momentum factor to identify "
        "highly active and trending stocks for entry. A breakout indicates the beginning of "
        "a new trend, while strong momentum suggests sustained price movement."
    )

    def __init__(self, window_breakout: int = 20, top_n_breakout: int = 5, window_momentum: int = 10) -> None:
        self._window_breakout = window_breakout
        self._top_n_breakout = top_n_breakout
        self._window_momentum = window_momentum

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_breakout)
        if closes.height < self._window_breakout:
            return Signal(information_available_at=stamp, weights={})

        breakout_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_breakout:
                continue
            if values[-1] >= max(values):
                breakout_picks.append(symbol)

        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history(lookback=self._window_momentum)
            momentum = (history[history["symbol"] == symbol]["close"].mean() - history[history["symbol"] == symbol]["open"].mean()) / history[history["symbol"] == symbol]["open"].std()
            if not momentum.is_nan():
                momentum_scores[symbol] = float(momentum)

        combined_scores: dict[str, float] = {symbol: 0.0 for symbol in view.symbols}
        for symbol in breakout_picks:
            combined_scores[symbol] += 1.0
        for symbol, score in momentum_scores.items():
            if symbol in combined_scores:
                combined_scores[symbol] += score

        top_symbols = sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)[:self._top_n_breakout]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest