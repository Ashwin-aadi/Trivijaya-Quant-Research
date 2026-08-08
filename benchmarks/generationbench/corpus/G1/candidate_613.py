from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their mean "
        "will revert to it. A short-horizon approach can identify recent overbought or oversold "
        "conditions and exploit them."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.is_empty() or history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window + 1)
        if any(col not in closes.columns for col in history["symbol"].to_list()):
            return Signal(information_available_at=stamp, weights={})

        symbol_returns: dict[str, float] = {}
        for symbol in history["symbol"].to_list():
            hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
            recent_close = view.latest_close()[symbol]
            mean_close = (
                hist.select(pl.col("adj_close").mean())
                .collect()
                .get(0)
                .to_series()
                .to_list()[0]
            )
            z_score = (recent_close - mean_close) / history["adj_close"].std().item()

            if z_score <= -1:
                symbol_returns[symbol] = 0.5
            elif z_score >= 1:
                symbol_returns[symbol] = -0.5

        non_empty_symbols = {s: w for s, w in symbol_returns.items() if w != 0}
        if not non_empty_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_sum = sum(non_empty_symbols.values())
        adjusted_weights = {s: w / weight_sum for s, w in non_empty_symbols.items()}
        return Signal(
            information_available_at=stamp,
            weights=adjusted_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest