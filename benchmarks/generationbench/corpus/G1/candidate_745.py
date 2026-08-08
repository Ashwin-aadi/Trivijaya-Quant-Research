from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion seeks to profit from prices returning to historical "
        "means. If a stock's price has deviated significantly from its mean over the past 20 days, "
        "it is expected to revert towards that mean."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean().alias("mean_close"))
            .collect()
            .get_column("mean_close")
            .item()
        )
        std_dev = (
            history.select(
                (pl.col("adj_close") - pl.col("adj_close").mean()).stddev().alias("std_dev")
            )
            .collect()
            .get_column("std_dev")
            .item()
        )

        if mean_close == 0 or std_dev == 0:
            return Signal(information_available_at=stamp, weights={})

        z_scores = (history["adj_close"] - mean_close) / std_dev
        candidates: list[str] = []

        for symbol in view.symbols:
            value = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()[-1]
            if abs(z_scores[symbol]) > self._threshold and z_scores[symbol].is_nan():
                continue

            if (
                (value < mean_close - std_dev * self._threshold and
                 z_scores[symbol] < 0)
                or
                (value > mean_close + std_dev * self._threshold and
                 z_scores[symbol] > 0)
            ):
                candidates.append(symbol)

        weight = 1.0 / len(candidates) if candidates else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest