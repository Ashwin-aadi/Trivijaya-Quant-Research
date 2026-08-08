from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MovingAverageAndStopLoss(Strategy):
    rationale = (
        "Combining a 50-day simple moving average to identify long-term trends with a stop-loss "
        "mechanism can help in managing risk. The moving average provides a basis for potential "
        "buy and sell signals while the stop-loss ensures that losses are contained within "
        "manageable levels."
    )

    def __init__(self, window: int = 50, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_closes: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            daily_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(daily_closes[-self._window:]) / self._window
            mean_closes[symbol] = mean_close

        # Filter symbols based on their proximity to the 50-day moving average
        picks: list[str] = [symbol for symbol, close in mean_closes.items() if close >= 1.02 * history[symbol][-1]]
        
        if len(picks) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest