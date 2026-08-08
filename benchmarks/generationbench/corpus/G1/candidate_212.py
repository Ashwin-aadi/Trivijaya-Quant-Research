from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can lead to continuation "
        "of the trend. By identifying symbols with significant price movements supported by volume, "
        "we aim to capture these opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            price_moves = history.select(
                pl.col("adj_close").filter(pl.col("symbol") == symbol).to_series().to_list()
            ).transpose()[0].drop_nulls()

            open_price = float(history.filter((pl.col("symbol") == symbol) & (pl.col("session_date") == stamp - date(2023, 1, 1))).select(pl.col("open")).item())
            close_price = float(price_moves[-1])
            volume_change = history.select(
                pl.col("volume").filter((pl.col("symbol") == symbol) & (pl.col("session_date") <= stamp - date(2023, 1, 1))).sum(),
                pl.col("volume").filter(pl.col("symbol") == symbol).sum()
            ).to_series().to_list()

            if close_price / open_price >= self._threshold or open_price / close_price >= self._threshold:
                signals.append(symbol)

        signals = signals[:5]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest