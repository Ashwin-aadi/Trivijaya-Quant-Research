from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines a momentum signal with a relative strength signal. Momentum "
        "suggests that stocks which have recently performed well will continue to do so, while "
        "relative strength identifies outperformers compared to the market. The combination of "
        "these two signals aims to identify strong and consistently performing stocks."
    )

    def __init__(self, momentum_window: int = 20, relative_strength_window: int = 30) -> None:
        self._momentum_window = momentum_window
        self._relative_strength_window = relative_strength_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._relative_strength_window)
        if history.height < self._relative_strength_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_history = [float(v) for v in history[symbol].select("close").to_list()[0]]
            if len(close_history) < self._momentum_window:
                continue
            momentum_score = (close_history[-1] - close_history[0]) / sum(close_history)
            momentum_scores[symbol] = momentum_score

        relative_strength_scores = {}
        for symbol in view.symbols:
            closes = [float(v) for v in history[symbol].select("adj_close").to_list()[0]]
            if len(closes) < self._relative_strength_window:
                continue
            performance_vs_market = (closes[-1] - closes[0]) / sum(closes)
            relative_strength_scores[symbol] = performance_vs_market

        combined_scores = {
            symbol: momentum_scores.get(symbol, 0) + relative_strength_scores.get(symbol, 0)
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
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest