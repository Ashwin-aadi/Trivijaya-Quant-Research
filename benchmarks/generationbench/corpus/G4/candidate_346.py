from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Leveraging calendar effects such as month-end buying pressure or earnings announcement periods, "
        "this strategy aims to capitalize on historical patterns of stock performance around significant dates. "
        "Behavioral biases and institutional practices contribute to these patterns."
    )

    def __init__(self, window_before: int = 10, window_after: int = 5) -> None:
        self._window_before = window_before
        self._window_after = window_after

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_before + self._window_after)

        if closes.height < self._window_before + self._window_after:
            return Signal(information_available_at=stamp, weights={})

        abnormal_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_before + self._window_after:
                continue

            returns = [
                float((values[i] / values[i - 1] - 1.0)) for i in range(1, len(values))
            ]
            abnormal_returns[symbol] = (
                sum(returns[-self._window_after:]) -
                sum(returns[-(self._window_before + self._window_after) : -(self._window_after)])
            )

        sorted_symbols = [
            s for _, s in sorted(abnormal_returns.items(), key=lambda item: abs(item[1]), reverse=True)
        ][:20]
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