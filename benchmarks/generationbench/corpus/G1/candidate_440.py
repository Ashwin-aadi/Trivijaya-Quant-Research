from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum. When a stock's price"
        "moves in a clear direction and is accompanied by higher volume, it suggests "
        "increased investor interest and potentially stronger future performance."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_risers, top_fallers = self._find_top_movers(history)

        weight = 1.0 / max(len(top_risers), len(top_fallers))
        weights: dict[str, float] = {}

        if top_risers:
            for symbol in top_risers:
                weights[symbol] = weight

        if not weights and history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )

    def _find_top_movers(self, history: pl.DataFrame) -> tuple[list[str], list[str]]:
        top_risers = []
        top_fallers = []

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            daily_returns = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .sort("session_date", descending=False)
            )

            if daily_returns.height < self._window:
                continue

            returns = [float(v) for v in daily_returns["return"].to_list()]
            volume_changes = history.filter(pl.col("symbol") == symbol).select(
                pl.col("session_date"), "volume"
            ).sort("session_date", descending=False)

            if volume_changes.height < self._window:
                continue

            volumes = [float(v) for v in volume_changes["volume"].to_list()]

            for i in range(self._window - 1, len(returns)):
                if all(
                    volumes[i - w] < volumes[i]
                    and returns[i - w] > returns[i]
                    for w in range(1, self._window)
                ):
                    top_fallers.append(symbol)
                    break

            for i in range(self._window - 1, len(returns)):
                if all(
                    volumes[i - w] < volumes[i]
                    and returns[i - w] < returns[i]
                    for w in range(1, self._window)
                ):
                    top_risers.append(symbol)
                    break

        return top_risers, top_fallers


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest