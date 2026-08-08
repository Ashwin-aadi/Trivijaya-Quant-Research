from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "The dual momentum strategy seeks to identify stocks that are both strong performers in the short-term and long-term. "
        "Stocks with both high recent returns and historically strong performance may indicate overbought conditions, suggesting a potential reversal."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_close = float(view.latest_close()[symbol])
            short_returns = (
                (recent_close - closes[symbol][-self._short_window :].mean())
                / closes[symbol][-self._short_window :].mean()
            )
            long_returns = (
                (recent_close - closes[symbol][-self._long_window :].mean())
                / closes[symbol][-self._long_window :].mean()
            )

            momentum_scores[symbol] = short_returns + long_returns

        sorted_symbols = [
            s for _, s in sorted(momentum_scores.items(), key=lambda item: item[1], reverse=True)
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest