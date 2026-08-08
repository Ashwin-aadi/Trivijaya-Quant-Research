from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate strong market sentiment and potential "
        "continuation of the trend. High volume at a significant price level suggests traders are "
        "confident in their move, which may lead to further gains or losses."
    )

    def __init__(self, window: int = 10, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            adj_close_series = pl.col(symbol)
            volume_series = pl.col(f"{symbol}_volume")

            close_change = (adj_close_series[-1] - adj_close_series[-2]) / adj_close_series[-2]
            high_volume = (
                volume_series.filter(adj_close_series == history[symbol].sort("session_date", descending=True).head(1)[0])
                 .sum()
                 .to_list()[0]
            )

            if close_change >= self._threshold and high_volume > 0:
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest