from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion against a trailing reference level suggests that after a price "
        "deviation from the mean, prices tend to revert back. This can be used to identify "
        "overbought or oversold conditions and generate trading signals."
    )

    def __init__(self, window: int = 50, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close = {symbol: float(v) for symbol, v in view.latest_close().items()}
        mean_price = history.select(pl.col("adj_close").mean()).item()
        trailing_reference = latest_close[view.as_of.strftime("%Y-%m-%d")] * self._threshold

        signals: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].drop_nulls().to_list()]
            latest_price = latest_close[symbol]
            mean_adj_close = sum(adj_closes[-self._window:]) / self._window

            if (latest_price > trailing_reference and mean_adj_close < trailing_reference) or (
                    latest_price < trailing_reference and mean_adj_close > trailing_reference):
                signals.append((symbol, 1.0 / len(view.symbols)))

        return Signal(information_available_at=stamp, weights=dict(signals))


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest