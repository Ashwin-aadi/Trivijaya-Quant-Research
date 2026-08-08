from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price levels that revert to a trailing average indicate mean reversion "
        "in the market. Inefficient markets may exhibit such patterns, and this strategy "
        "aims to profit by buying underpriced stocks and selling overpriced ones."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trailing average for each symbol
        avg_close = history.group_by("symbol").agg(
            (pl.col("adj_close").mean()).alias("trailing_avg")
        ).select(["symbol", "trailing_avg"])

        # Join with the latest closes to calculate deviations from the mean
        joined_history = history.join(avg_close, on="symbol", how="inner")
        deviations = (
            joined_history.with_columns(
                (pl.col("adj_close") - pl.col("trailing_avg")).alias("deviation")
            )
            .sort("session_date")
            .tail(1)
        )

        # Find symbols with significant positive and negative deviations
        positive_deviations = [float(v) for v in deviations[deviations["deviation"] > 0]["adj_close"].to_list()]
        negative_deviations = [float(v) for v in deviations[deviations["deviation"] < 0]["adj_close"].to_list()]

        if not positive_deviations and not negative_deviations:
            return Signal(information_available_at=stamp, weights={})

        weight_pos = len(positive_deviations) / (len(positive_deviations) + len(negative_deviations))
        weight_neg = len(negative_deviations) / (len(positive_deviations) + len(negative_deviations))

        positive_weights = {s: weight_pos for s in [dev.to_dict()["symbol"] for dev in positive_deviations]}
        negative_weights = {s: -weight_neg for s in [dev.to_dict()["symbol"] for dev in negative_deviations]}

        weights = {**positive_weights, **negative_weights}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest