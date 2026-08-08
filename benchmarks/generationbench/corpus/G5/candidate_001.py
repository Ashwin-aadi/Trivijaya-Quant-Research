from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices and financial market indices will eventually "
        "reverse after a significant deviation from the average. For short horizons, sudden price "
        "movements are likely to revert back towards historical means over a 10-day period."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.mean().height == 1
            and float(closes.mean().to_series())
            or None
        )
        if mean_close is None:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            z_score = (values[-1] - mean_close) / pl.col("adj_close").std().over(pl.arange(0, self._window))
            if abs(z_score) > self._threshold:
                picks.append(symbol)

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