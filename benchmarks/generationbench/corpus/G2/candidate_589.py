from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion exploits mean-reverting behavior in stock prices. If a stock has "
        "underperformed its peers for several days, it is expected to revert back to the "
        "mean, providing an opportunity for profit."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            window_mean = sum(values[-self._window:]) / self._window
            score = abs((values[-1] - window_mean) / window_mean)
            mean_reversion_scores[symbol] = score

        ranked_symbols = sorted(mean_reversion_scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbol = ranked_symbols[0][0]
        weight = 1.0
        return Signal(
            information_available_at=stamp, weights={top_symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest