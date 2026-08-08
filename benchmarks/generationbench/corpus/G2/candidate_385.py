from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionShortHorizon(Strategy):
    rationale = (
        "Short-horizon mean reversion is based on the economic belief that asset prices will "
        "tend to revert to their historical average over a short period. When an asset price"
        " deviates significantly from its moving average, it has historically tended to move back towards it."
    )

    def __init__(self, window: int = 5, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window or any(
            symbol not in closes.columns for symbol in view.symbols
        ):
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes.drop_nulls().group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation"))
        )

        filtered_symbols = [
            symbol
            for symbol in view.symbols
            if mean_close.filter(
                (pl.col("symbol") == symbol) & (
                    (pl.col("deviation") / self._threshold > 1)
                )
            ).height > 0
        ]

        weights: dict[str, float] = {}
        for symbol in filtered_symbols:
            weight = 1.0 / len(filtered_symbols)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest