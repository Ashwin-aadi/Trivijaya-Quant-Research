from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of financial assets to return to their "
        "historical mean. In an efficient market, extreme price movements are temporary, and "
        "prices will revert towards a long-term average. By identifying stocks that have moved "
        "away from this average in a short window, we can profit when they revert."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_prices: dict[str, float] = (
            history.group_by("symbol").agg(pl.col("adj_close").mean().alias("m")).to_dict(False)
        )
        latest_closes = {row["symbol"]: row["adj_close"] for row in view.closes(lookback=self._window).rows()}
        
        candidates: list[str] = []
        for symbol, mean_price in mean_prices.items():
            if symbol not in latest_closes:
                continue
            current_price = latest_closes[symbol]
            if abs(current_price - mean_price) / mean_price > 0.15:
                candidates.append(symbol)

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest