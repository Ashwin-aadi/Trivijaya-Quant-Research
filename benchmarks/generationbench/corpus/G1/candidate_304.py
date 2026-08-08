from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Prices that revert to the mean of a trailing period are often good entry points. "
        "This strategy identifies symbols where recent prices have deviated significantly from their trailing average."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        symbols = history["symbol"].unique().to_list()
        signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in symbols:
                continue

            df = history.filter(pl.col("symbol") == symbol)
            close_prices = df.select("adj_close").to_numpy().flatten()

            mean_price = sum(close_prices) / len(close_prices)
            latest_close = float(view.latest_close()[symbol])

            deviation = abs(latest_close - mean_price)
            if deviation / mean_price >= self._threshold:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in signals.keys()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_numpy()[0][0]
    assert isinstance(newest, date)
    return newest