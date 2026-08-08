from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets that have deviated "
        "significantly from their trailing average price, we can exploit this tendency."
    )

    def __init__(self, window: int = 50, zscore_threshold: float = 2.0) -> None:
        self._window = window
        self._zscore_threshold = zscore_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().plike("adj_close").item()
        std_close = (closes.std().plike("adj_close")).item()

        zscores = [
            (float(c) - mean_close) / std_close for _, c in closes.to_dict(as_pandas=False).items()
        ]

        signals: list[str] = []
        for symbol, zscore in zip(view.symbols, zscores):
            if abs(zscore) >= self._zscore_threshold:
                signals.append(symbol)

        signals = signals[:5]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest