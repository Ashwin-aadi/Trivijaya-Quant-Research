from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of stock prices to return to their mean "
        "over time. By identifying stocks that have deviated significantly from their historical "
        "mean price, we can expect a reversal in the near future."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol").agg(pl.col("adj_close").mean().alias("mean"))
        )
        latest_closes = view.closes()
        symbols = [s for s in view.symbols if s in latest_closes.columns]

        signal_symbols: list[str] = []
        for symbol in symbols:
            mean_price = float(mean_close[mean_close["symbol"] == symbol]["mean"].to_list()[0])
            current_price = float(latest_closes[symbol].to_list()[-1])

            if abs(current_price - mean_price) / mean_price >= self._threshold:
                signal_symbols.append(symbol)

        if not signal_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signal_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signal_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest