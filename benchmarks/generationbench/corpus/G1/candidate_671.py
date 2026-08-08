from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendVolatility(Strategy):
    rationale = (
        "This strategy identifies stocks with a trending price movement and low volatility. "
        "A stock that is trending up but not experiencing high volatility may be an ideal candidate for a breakout or continuation of the trend."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window * 2 + 1:
                continue

            # Calculate the rolling mean and standard deviation
            mean_values = (pl.DataFrame({"close": values})
                           .rolling_mean(window_size=self._window)
                           .select([("close", "mean")])
                           .to_series()
                           .to_list())
            std_values = (pl.DataFrame({"close": values})
                          .rolling_std(window_size=self._window)
                          .select([("close", "std")])
                          .to_series()
                          .to_list())

            # Calculate the rolling returns
            returns = [(v / prev - 1.0) for v, prev in zip(values[1:], values[:-1])]
            if abs(returns[-1]) > 1.5 * std_values[-1]:
                picks.append(symbol)

        picks = picks[: self._top_n]
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