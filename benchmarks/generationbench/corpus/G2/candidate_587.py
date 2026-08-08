from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "Combining short-term and long-term momentum can capture both the trend-following "
        "effectiveness of shorter time frames and the smoothing benefits of longer ones. This "
        "approach aims to identify stocks with strong relative performance over multiple periods."
    )

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._short_window)
        if closes.height < self._short_window:
            return Signal(information_available_at=stamp, weights={})

        short_momentum: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._short_window:
                continue
            latest_close = values[-1]
            rank = (pl.Series(values).rank(method="ordinal", descending=True)).item()
            short_momentum[symbol] = 1 - (rank / (self._short_window + 1))

        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        long_momentum: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._long_window:
                continue
            latest_close = values[-1]
            rank = (pl.Series(values).rank(method="ordinal", descending=True)).item()
            long_momentum[symbol] = 1 - (rank / (self._long_window + 1))

        combined_ranking: dict[str, float] = {}
        for symbol in short_momentum.keys():
            if symbol not in long_momentum:
                continue
            combined_ranking[symbol] = (
                short_momentum[symbol] + long_momentum[symbol]
            )

        top_symbols = sorted(combined_ranking.items(), key=lambda x: -x[1])[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [sym for sym, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest