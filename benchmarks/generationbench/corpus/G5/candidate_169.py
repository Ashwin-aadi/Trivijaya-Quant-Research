from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that prices will revert to the mean after "
        "deviating from it. By identifying symbols that have moved significantly away from "
        "their recent average price and are currently below or above a threshold, we can "
        "profit from their expected return to normal levels."
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
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).abs().alias("deviation")
            )
            .sort("deviation", descending=True)
        )

        candidates: list[str] = []
        for symbol in mean_close["symbol"].to_list():
            if (
                float(history.filter(pl.col("symbol") == symbol)["adj_close"].last())
                - self._threshold * mean_close.filter(pl.col("symbol") == symbol)[
                    "mean"
                ].item()
            ) < 0:
                candidates.append(symbol)
            elif (
                self._threshold * mean_close.filter(pl.col("symbol") == symbol)[
                    "mean"
                ].item()
                - float(history.filter(pl.col("symbol") == symbol)["adj_close"].last())
            ) > 0:
                candidates.append(symbol)

        weight = 1.0 / len(candidates)
        if not candidates:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest