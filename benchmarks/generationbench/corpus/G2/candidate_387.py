from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are often considered to be more efficient in terms of trading and "
        "execution costs. By investing equally across the most liquid stocks, we aim to capture "
        "the benefits of reduced slippage and potentially higher trading volumes."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            volume_history = [float(v) for v in history[symbol]["volume"].drop_nulls().to_list()]
            if len(volume_history) < self._window:
                continue

            # Calculate the mean trading volume over the window period
            mean_volume = sum(volume_history[-self._window:]) / min(self._window, len(volume_history))
            liquidity_scores[symbol] = mean_volume

        sorted_symbols = [s for s in sorted(liquidity_scores, key=liquidity_scores.get, reverse=True)]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols[:5]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest