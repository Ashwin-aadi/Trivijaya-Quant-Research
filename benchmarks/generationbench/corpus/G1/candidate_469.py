from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion looks for stocks that have recently deviated from their "
        "historical price levels and are likely to revert. This strategy exploits the tendency of"
        "asset prices to return to an average value over time."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
        )

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in mean_close.columns or mean_close.height < self._window + 1:
                continue
            latest_deviation = float(mean_close[stamp, f"{symbol}_deviation"])
            if latest_devoration > self._threshold:
                signals.append(symbol)

        weights: dict[str, float] = {}
        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        for symbol in signals:
            weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest