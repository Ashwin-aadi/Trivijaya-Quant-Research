from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels revert to their trailing mean. This suggests that after a period of "
        "abnormal performance (either positive or negative), the price will return to its "
        "historical average."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_closes = [float(v) for v in history["adj_close"].drop_nulls().to_list()]
        mean_price = sum(recent_closes) / len(recent_closes)
        symbols = view.symbols

        def compute_signal(symbol: str) -> float:
            symbol_history = history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].to_list()
            if not symbol_history:
                return 0.0
            recent_close = float(symbol_history[-1])
            z_score = (recent_close - mean_price) / max(1e-6, abs(recent_close - mean_price))
            return 1.0 if z_score > 2 else (-1.0 if z_score < -2 else 0.0)

        weights: dict[str, float] = {symbol: compute_signal(symbol) for symbol in symbols}
        weight_sum = sum(weights.values())
        adjusted_weights = {k: v / max(1e-6, weight_sum) for k, v in weights.items()}
        return Signal(information_available_at=stamp, weights=adjusted_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest