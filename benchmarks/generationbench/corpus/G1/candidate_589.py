from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion aims to identify stocks that have deviated significantly "
        "from their historical average and are likely to revert. This strategy exploits the "
        "tendency of prices to return to a long-term average."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)  # Include extra days for mean calculation
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close").exclude("session_date"))
        means = (closes.mean().to_dict()["adj_close"]).items()
        std_devs = (closes.std().to_dict()["adj_close"]).items()

        signals: dict[str, float] = {}
        for symbol in symbols:
            latest_close = view.latest_close()[symbol]
            mean = next(iter(means))[1][symbol]
            std_dev = next(iter(std_devs))[1][symbol]

            z_score = (latest_close - mean) / std_dev
            if abs(z_score) > 2.0:  # Consider stocks with extreme z-scores for reversion
                signals[symbol] = 1.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest