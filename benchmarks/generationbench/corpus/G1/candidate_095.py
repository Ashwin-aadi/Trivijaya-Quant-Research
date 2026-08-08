from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines short-term and long-term momentum to identify "
        "stocks that are both gaining ground in the recent past and have a higher"
        "trend over a longer period. Such stocks are likely to continue their upward trend."
    )

    def __init__(self, window_short: int = 10, window_long: int = 60) -> None:
        self._window_short = window_short
        self._window_long = window_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_long)

        if closes.height < self._window_long or closes.height < self._window_short:
            return Signal(information_available_at=stamp, weights={})

        short_momentum: dict[str, float] = {}
        long_momentum: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_long or len(values) < self._window_short:
                continue

            short_returns = (values[-1] / values[-self._window_short - 1]) - 1.0
            long_returns = (values[-1] / values[-self._window_long - 1]) - 1.0

            if short_returns > 0 and long_returns > 0:
                short_momentum[symbol] = short_returns
                long_momentum[symbol] = long_returns

        sorted_short = sorted(short_momentum.items(), key=lambda x: x[1], reverse=True)
        sorted_long = sorted(long_momentum.items(), key=lambda x: x[1], reverse=True)

        picks: list[str] = []
        for symbol, _ in sorted_short[:5]:
            if symbol not in long_momentum or (symbol in long_momentum and short_momentum[symbol] > long_momentum[symbol]):
                picks.append(symbol)

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