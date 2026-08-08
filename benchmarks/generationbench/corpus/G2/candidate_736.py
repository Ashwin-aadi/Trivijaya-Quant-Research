from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price reversion occurs when a stock that has deviated significantly from its historical "
        "price levels tends to return towards those levels. A trailing reference allows us to "
        "identify overbought or oversold conditions and capitalize on mean-reverting behavior."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            adj_closes = [float(v) for v in history.select(pl.col(symbol)).to_series().drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            mean_price = sum(adj_closes[-20:]) / 20.0
            current_close = view.latest_close()[symbol]
            reversion_score = abs(current_close - mean_price)

            reversion_scores[symbol] = reversion_score

        symbols_to_trade = sorted(reversion_scores, key=reversion_scores.get)[:10]

        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={(symbol): weight for symbol in symbols_to_trade},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest