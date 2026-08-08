from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines momentum and volatility to capture both trending markets "
        "and periods of high market activity. Momentum suggests that stocks with rising prices "
        "tend to continue rising, while high volatility can indicate upcoming price movements."
    )

    def __init__(self, momentum_window: int = 20, volatility_window: int = 10) -> None:
        self._momentum_window = momentum_window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._momentum_window, self._volatility_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = set(view.symbols).intersection(history.columns)

        momentum_scores: dict[str, float] = {}
        volatility_scores: dict[str, float] = {}

        for symbol in symbols:
            symbol_history = history.select(["session_date", f"{symbol}"])
            if symbol_history.height < self._momentum_window + 1:
                continue

            closes = [float(v) for v in symbol_history[symbol].to_list()]
            momentum_score = (closes[-1] - closes[0]) / sum(closes)
            volatility_score = max([abs(closes[i] - closes[i-1]) for i in range(1, len(closes))])

            momentum_scores[symbol] = momentum_score
            volatility_scores[symbol] = volatility_score

        combined_scores = {
            symbol: 0.5 * momentum_scores[symbol] + 0.5 * volatility_scores[symbol]
            for symbol in symbols
        }

        sorted_symbols = [
            s[0] for s in sorted(combined_scores.items(), key=lambda x: -x[1])
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