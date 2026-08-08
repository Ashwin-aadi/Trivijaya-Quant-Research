from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum can help in identifying stocks that "
        "are consistently performing well. Short-term momentum captures immediate market "
        "reactions, while long-term momentum reflects sustained performance."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        short_moments: dict[str, float] = {}
        long_moments: dict[str, float] = {}

        for symbol in view.symbols:
            adj_close = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(adj_close) < self._short_window + 1:
                continue
            short_moment = (adj_close[-1] - adj_close[0]) / max(adj_close)
            long_moment = (adj_close[-1] - adj_close[self._long_window // 2]) / max(adj_close)
            if not pl.all(pl.col(symbol).is_nan()):
                short_moments[symbol] = short_moment
                long_moments[symbol] = long_moment

        combined_moments = {
            symbol: (short_moments.get(symbol, 0) + long_moments.get(symbol, 0)) / 2.0
            for symbol in view.symbols
        }

        sorted_symbols = [
            k for k, _ in sorted(combined_moments.items(), key=lambda item: -item[1])
        ]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest