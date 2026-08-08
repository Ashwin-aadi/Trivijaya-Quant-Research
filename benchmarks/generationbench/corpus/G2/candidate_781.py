from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that asset prices will revert to the mean "
        "after a significant deviation from it. This phenomenon can be observed when prices "
        "move too far in one direction due to market noise or temporary factors, and then "
        "eventually return to their historical average."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_signal: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue

            daily_returns = (history.select(pl.col("adj_close").reverse())["adj_close"].to_list()[1:] /
                             history.select(pl.col("adj_close").reverse())["adj_close"].to_list()[:-1] - 1)
            mean_return = sum(daily_returns) / len(daily_returns)
            std_dev = (sum((x - mean_return) ** 2 for x in daily_returns) / len(daily_returns)) ** 0.5

            if std_dev == 0:
                continue

            z_score = (history.select(pl.col("adj_close").reverse())["adj_close"].to_list()[-1] -
                       mean_return) / std_dev
            if z_score < -2:
                mean_reversion_signal[symbol] = 1.0

        if not mean_reversion_signal:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(mean_reversion_signal)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in mean_reversion_signal.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest