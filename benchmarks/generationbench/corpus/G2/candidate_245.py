from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts are often followed by continuation patterns. After a strong breakout, the "
        "price may continue in the direction of the breakout, providing an opportunity for long-term "
        "profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).filter(
            (pl.col("session_date") >= pl.col("session_date").max().shift(-self._window))
            & (pl.col("session_date") < pl.col("session_date").max())
        )
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            close_price = float(view.latest_close()[symbol])
            last_session_date = history.select(pl.col("session_date").max()).row(0)[0]
            price_history = history.filter(
                pl.col("symbol") == symbol
            ).select(["session_date", "close"])
            if price_history.height < self._window:
                continue

            breakout_price = (
                price_history.sort(by="session_date")
                .select(pl.col("close").filter(pl.col("session_date") < last_session_date))
                .tail(1)
                .to_dict(as_series=False)[0][0]
            )
            if close_price > breakout_price * self._threshold:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest