from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies aim to capitalize on mean-reverting behavior. "
        "After a period of strong movement in one direction, prices often revert to their "
        "historical means. This strategy identifies symbols that have moved too far from "
        "their 20-day moving average and expects a reversal."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("adj_close").mean().alias("m")
        ).select(pl.col("m")).to_series()[0]

        recent_closes = view.closes(lookback=self._window)
        if recent_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        reversion_signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in recent_closes.columns:
                continue
            values = [float(v) for v in recent_closes[symbol].drop_nulls().to_list()]
            if abs(values[-1] - mean_close) / mean_close > 0.2:
                reversion_signals.append(symbol)

        weight = 1.0 / len(reversion_signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in reversion_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest