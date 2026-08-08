from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when a stock's price returns to its historical average. "
        "By identifying stocks that have deviated significantly from their mean, we can predict "
        "a future return to the mean, generating profitable trades."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        means: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            mean_price = sum(prices[-self._window:]) / self._window
            means[symbol] = mean_price

        deviations: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in means or symbol not in history.columns:
                continue
            latest_close = float(history[history["symbol"] == symbol]["adj_close"].max())
            deviation = (latest_close - means[symbol]) / mean_price
            deviations[symbol] = deviation

        sorted_symbols = [
            s for s, d in sorted(deviations.items(), key=lambda item: abs(item[1]), reverse=True) if abs(d) > self._threshold
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