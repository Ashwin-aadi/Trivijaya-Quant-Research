from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionStrategy(Strategy):
    rationale = (
        "Leveraging mean-reverting characteristics of stock prices to capitalize on deviations "
        "from historical price levels. This strategy aims to buy undervalued stocks and sell "
        "overvalued ones based on moving averages."
    )

    def __init__(self, short_window: int = 20, long_window: int = 50, threshold: float = 0.05) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].unique().to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_short = (
            history.filter(pl.col("symbol").is_in(symbols))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._short_window - 1) - 1.0)
                .mean()
                .alias("mean_short_return"),
            )
        )["mean_short_return"].to_list()

        mean_long = (
            history.filter(pl.col("symbol").is_in(symbols))
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._long_window - 1) - 1.0)
                .mean()
                .alias("mean_long_return"),
            )
        )["mean_long_return"].to_list()

        mean_reversion_scores = [
            abs(mean_short[i] + mean_long[i]) for i in range(len(symbols))
        ]

        picks: list[str] = []
        for symbol, score in zip(symbols, mean_reversion_scores):
            if score > self._threshold:
                picks.append(symbol)

        weight = 1.0 / len(picks) if picks else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_python()
    assert isinstance(newest, date)
    return newest