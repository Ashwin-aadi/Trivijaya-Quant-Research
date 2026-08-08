from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies exploit mean-reverting tendencies in stock prices. "
        "Historically, stocks that have recently deviated from their long-term means are "
        "likely to return towards those levels. This strategy identifies such price deviations"
        " and bets on a reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in hist["adj_close"].to_list()]
            mean_price = sum(prices[-20:]) / 20
            latest_price = float(hist.select(pl.last("adj_close")).item())
            deviation = abs(latest_price - mean_price)
            reversion_scores[symbol] = deviation

        sorted_symbols = [
            s for s in view.symbols if s in reversion_scores and reversion_scores[s]
        ]
        top_n_symbols = sorted_symbols[:5]

        weights = {s: 1.0 / len(top_n_symbols) for s in top_n_symbols}
        return Signal(
            information_available_at=stamp, weights={s: weights[s] for s in view.symbols if s in weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest