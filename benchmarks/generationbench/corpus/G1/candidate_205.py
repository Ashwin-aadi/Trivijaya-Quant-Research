from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their "
        "historical average are likely to return to that mean. By identifying such deviations "
        "in the short term (10 days), we can generate profitable trades."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(pl.col("adj_close").mean().alias("mean")).height > 0
        mean_close = float(closes.filter(mean_close)["mean"].item())

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = view.latest_close()[symbol]
            z_score = (latest_close - mean_close) / (closes[symbol].std().item() or 1)
            if abs(z_score) > self._threshold and latest_close != mean_close:
                signals[symbol] = -0.1 if z_score < 0 else 0.1

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / sum(signals.values())
        weighted_signals = {s: w * weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, weights=weighted_signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest