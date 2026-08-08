from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy combines two simple momentum indicators: the 10-day return and the "
        "20-day return. Stocks with both positive returns are expected to continue outperforming."
    )

    def __init__(self, window_short: int = 10, window_long: int = 20) -> None:
        self._window_short = window_short
        self._window_long = window_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_short, self._window_long))
        if history.height < max(self._window_short, self._window_long):
            return Signal(information_available_at=stamp, weights={})

        short_returns: list[str] = []
        long_returns: list[str] = []

        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            close_values = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            if len(close_values) < max(self._window_short, self._window_long):
                continue

            short_return = (close_values[-1] / close_values[-self._window_short - 1] - 1.0) * 100
            long_return = (close_values[-1] / close_values[-self._window_long - 1] - 1.0) * 100

            short_returns.append(str(short_return))
            long_returns.append(str(long_return))

        picks: list[str] = []
        for symbol in view.symbols:
            if float(short_returns[view.symbols.index(symbol)]) > 0 and \
                    float(long_returns[view.symbols.index(symbol)]) > 0:
                picks.append(symbol)

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