from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies exploit deviations from the mean price. "
        "Short-horizon mean reversion looks for stocks that have deviated significantly from their recent price range."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean().alias("mean"))
            .group_by("symbol")
            .agg(["mean"])
            .lazy()
            .collect()["mean"]
            .to_list()
        )

        deviations = [
            (float(history[symbol]["adj_close"].max()) - float(mean_close[i])),
            (float(mean_close[i]) - float(history[symbol]["adj_close"].min())),
        ]

        candidates = []
        for symbol, dev in zip(symbols, deviations):
            if (
                abs(dev[0]) > 2 * history.select(pl.col("adj_close").std().alias("std")).group_by("symbol").agg(["std"]).lazy().collect()["std"][i]
                or
                abs(dev[1]) > 2 * history.select(pl.col("adj_close").std().alias("std")).group_by("symbol").agg(["std"]).lazy().collect()["std"][i]
            ):
                candidates.append(symbol)

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest