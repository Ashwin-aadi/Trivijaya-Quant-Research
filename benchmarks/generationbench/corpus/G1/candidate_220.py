from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts towards historical means; stocks that have moved far from their "
        "historical price levels are expected to move back. This strategy identifies such "
        "over-moved stocks and allocates capital accordingly."
    )

    def __init__(self, window: int = 60, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_to_stats = (
            history.group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("mean"),
                (pl.col("adj_close") - pl.col("adj_close").mean()).std()
                * 2.0
                .alias("std_dev"),
            )
            .collect()
        )

        closes = view.closes(lookback=self._window)
        symbol_to_z_score = {
            symbol: float((closes[symbol][-1] - stat["mean"]) / stat["std_dev"])
            for symbol, stat in zip(closes.columns, symbol_to_stats.to_dict())
        }

        symbols_within_threshold = [
            symbol
            for symbol, z_score in symbol_to_z_score.items()
            if abs(z_score) > self._z_score_threshold
        ]

        if not symbols_within_threshold:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_within_threshold)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in symbols_within_threshold},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest