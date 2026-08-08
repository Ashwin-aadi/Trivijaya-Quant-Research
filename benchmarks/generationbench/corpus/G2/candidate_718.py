from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Seasonality can lead to predictable patterns in stock prices due to factors such as "
        "company-specific events (e.g., earnings releases), macroeconomic events (e.g., fiscal "
        "year ends), or investor behavior. For example, some stocks may consistently outperform"
        " during specific months of the year."
    )

    def __init__(self, window: int = 365, seasonality_periods: tuple[int, ...] = (120,)) -> None:
        self._window = window
        self._seasonality_periods = seasonality_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Extract the latest close prices for each symbol
        closes = view.closes(lookback=self._window)

        # Compute returns
        returns: pl.DataFrame = (
            closes.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._seasonality_periods[0]) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .select(["symbol", "return"])
        )

        # Filter out symbols with no return data
        returns = returns.filter(pl.col("return").is_not_null())

        # Calculate mean returns for each symbol
        mean_returns = (
            returns.group_by("symbol")
            .agg(
                pl.col("return").mean().alias("avg_return"),
                (pl.col("return") > 0).sum().alias("pos_count"),
                (pl.col("return") < 0).sum().alias("neg_count"),
            )
        )

        # Identify symbols with positive average returns
        top_symbols = mean_returns.filter(
            (pl.col("avg_return").is_not_null()) & (pl.col("pos_count") > pl.col("neg_count"))
        )["symbol"].to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest