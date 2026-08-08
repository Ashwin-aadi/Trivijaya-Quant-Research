from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies seek to exploit temporary deviations from the mean price. "
        "In the short term, prices that have deviated significantly from their average tend to "
        "return towards their long-term mean."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().item()
        symbols_to_trade = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            z_score = (latest_close - mean_close) / closes.select(pl.col(symbol)).std().item()
            if abs(z_score) > 1.5:  # Consider trading when z-score is more than 1.5 or less than -1.5
                symbols_to_trade.append(symbol)

        weights = {symbol: 0.2 for symbol in symbols_to_trade} if symbols_to_trade else {}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest