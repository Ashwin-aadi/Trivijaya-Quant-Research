from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies capitalize on the tendency of stock prices to revert "
        "to their historical average. By identifying stocks that have deviated significantly "
        "from this average in a short horizon, we can generate profitable trading signals."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean().alias("mean"))
            .select("mean")
            .to_series()
            .item()
        )

        recent_closes = view.closes(lookback=self._window)
        symbols = [symbol for symbol in view.symbols if symbol in recent_closes.columns]
        
        signals: list[str] = []
        for symbol in symbols:
            recent_prices = recent_closes[symbol].drop_nulls().to_list()
            if len(recent_prices) < self._window:
                continue
            latest_price = float(recent_prices[-1])
            if abs(latest_price - mean_close) > 3 * pl.col("adj_close").std().item():
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest