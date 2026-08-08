from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks where the current closing price is significantly below "
        "their 20-day simple moving average (SMA). Entering long positions at these points takes "
        "advantage of mean reversion tendencies in stock prices."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_20 = (closes["adj_close"].rolling_mean(self._window)).to_list()
        z_scores = [
            (close - sma) / close for close, sma in zip(closes["adj_close"].to_list(), sma_20)
        ]

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            z_score = float(z_scores[closes[symbol].height - 1])
            if z_score <= -1.0:
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