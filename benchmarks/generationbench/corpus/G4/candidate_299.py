from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy leverages a composite of macroeconomic and micro-specific metrics to "
        "identify companies with favorable market conditions. By combining GDP growth trends "
        "with earnings surprises, we aim to capitalize on both broad market sentiments and specific company performance."
    )

    def __init__(self, window: int = 180, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        macro_signal = (
            (history["adj_close"] / history["adj_close"].shift(self._window) - 1.0).mean()
            - self._threshold
        )

        micro_signals = {}
        for symbol in view.symbols:
            closes = view.closes(lookback=30)
            if symbol not in closes.columns:
                continue

            earnings = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(earnings) < 30:
                continue
            score = sum(float(c > 0) for c in (np.diff(earnings) / np.abs(np.diff(earnings)) - 1))
            micro_signals[symbol] = score

        ranked_composite_scores = [
            (
                symbol,
                macro_signal * (micro_signals.get(symbol, 0) or 0),
            )
            for symbol in view.symbols
        ]
        ranked_composite_scores.sort(key=lambda x: x[1], reverse=True)

        if not ranked_composite_scores:
            return Signal(information_available_at=stamp, weights={})

        num_investments = min(30, len(ranked_composite_scores))
        weight_per_company = 1.0 / num_investments
        selected_symbols = [s[0] for s in ranked_composite_scores[:num_investments]]

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight_per_company for symbol in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest