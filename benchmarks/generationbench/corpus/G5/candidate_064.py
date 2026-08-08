from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their "
        "trailing average price and are likely to revert. It focuses on mean reversion "
        "principles within the Indian market."
    )

    def __init__(self, window: int = 50, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbols = set(latest_closes.columns) - {"session_date"}

        signals: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history.symbol.to_list():
                continue
            adj_closes = (
                history.filter(pl.col("symbol") == symbol)["adj_close"]
                .to_list()
                .drop_nulls()
            )
            mean_adj_close = sum(adj_closes) / len(adj_closes)
            latest_close = float(latest_closes[symbol].item())

            reversion_signal = (latest_close - mean_adj_close) / mean_adj_close
            if abs(reversion_signal) > self._threshold:
                signals[symbol] = 1.0

        picks = [s for s, w in signals.items() if w == 1.0]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest