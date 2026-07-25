"""Hold names whose short moving average sits above their long moving average."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class SmaCrossover(Strategy):
    """Classic trend filter on two simple moving averages."""

    rationale = (
        "A short average above a long one indicates recent prices are running ahead of the "
        "established level, which is the conventional definition of an uptrend. The portfolio "
        "holds only names in that state and stays out of the rest."
    )

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._long)
        if closes.height < self._long:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = closes[symbol].drop_nulls().to_list()
            if len(values) < self._long:
                continue
            short_avg = sum(values[-self._short:]) / self._short
            long_avg = sum(values[-self._long:]) / self._long
            if short_avg > long_avg:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
