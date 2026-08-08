from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion2d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of financial prices to revert "
        "to their historical mean. By identifying assets that have deviated significantly "
        "from their average price, we can generate trade signals."
    )

    def __init__(self, window: int = 2, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        means: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_value = sum(values[-self._window:]) / self._window
            means[symbol] = mean_value

        deviations: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in means.keys():
                continue
            latest_close = float(view.latest_close()[symbol])
            deviations[symbol] = abs(latest_close - means[symbol])

        sorted_symbols = [k for k, v in sorted(deviations.items(), key=lambda item: item[1])]
        picks = sorted_symbols[: self._top_n]
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