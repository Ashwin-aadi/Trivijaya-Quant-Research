from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of stock prices to revert to their mean "
        "over short periods. This strategy identifies stocks that have significantly deviated from "
        "their recent mean price and bets on a return to the mean."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_price = sum(closes) / len(closes)
        z_scores = [(price - mean_price) / (max(closes) - min(closes)) for price in closes]
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_closes = [float(v) for v in history[symbol].to_list()]
            z_score = (symbol_closes[-1] - mean_price) / (max(symbol_closes) - min(symbol_closes))
            if abs(z_score) > self._threshold:
                picks.append(symbol)

        picks = picks[:5]
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