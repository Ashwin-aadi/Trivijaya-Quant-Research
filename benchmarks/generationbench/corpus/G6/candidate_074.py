from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySMA50(Strategy):
    rationale = (
        "This strategy captures trending movements in stock prices while adjusting risk exposure based on market volatility."
    )

    def __init__(self, window: int = 20, top_n: int = 15) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 49)
        if history.is_empty() or history.height < self._window + 49:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = history[symbol].drop_nulls()["adj_close"]
            returns = (close_series / close_series.shift(1) - 1.0).to_list()[1:]
            sma_50 = sum(close_series[-50:]) / 50
            volatility = pl.Series(returns).std()

            if close_series[-1] > sma_50 and volatility < history["adj_close"].std():
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